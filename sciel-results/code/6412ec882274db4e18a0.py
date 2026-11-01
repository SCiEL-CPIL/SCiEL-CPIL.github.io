import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _try_rank31_direct(n, m, p, U32, V32, W32, time_budget):
    """Try to find a rank-31 decomposition directly."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
                
    # Mask for non-zero entries to weight them higher
    # Actually, we just minimize sum of squares of all entries.
    # The non-zero entries are 1, others 0.
    
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Initialize with first 31 columns of rank-32 solution
    # Also try random initializations
    initializations = []
    initializations.append((U32[:, :rank].copy(), V32[:, :rank].copy(), W32[:, :rank].copy()))
    
    # Add a few random perturbations
    for i in range(5):
        U_rand = U32[:, :rank].copy() + np.random.randn(nm, rank) * 0.1
        V_rand = V32[:, :rank].copy() + np.random.randn(mp, rank) * 0.1
        W_rand = W32[:, :rank].copy() + np.random.randn(pn, rank) * 0.1
        initializations.append((U_rand, V_rand, W_rand))
        
    for U0, V0, W0 in initializations:
        if time.time() - start_time > time_budget:
            break
            
        def objective_and_grad(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = M - T_true
            
            # Gradient
            # grad_U[i,r] = 2 * sum_jk R[i,j,k] V[j,r] W[k,r]
            A = np.einsum('ijk,kr->ijr', R, W)
            grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
            
            # grad_V[j,r] = 2 * sum_ik R[i,j,k] U[i,r] W[k,r]
            B = np.einsum('ijk,kr->ijr', R, W) # Same as A
            grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
            
            # grad_W[k,r] = 2 * sum_ij R[i,j,k] U[i,r] V[j,r]
            D = np.einsum('ir,jr->ijr', U, V)
            grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
            
            sq_res = np.sum(R**2)
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return sq_res, grad
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 5000})
        except Exception:
            continue
            
        U_flat = result.x[:nm * rank]
        V_flat = result.x[nm * rank : nm * rank + mp * rank]
        W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        res = _compute_residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U, V, W
            
        if res < 1e-9:
            return U, V, W, res
            
    if best_res < 1e-9:
        return best_U, best_V, best_W, best_res
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using direct optimization
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 20, 100)
        result = _try_rank31_direct(n, m, p, U32, V32, W32, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-9:
                return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
                
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
