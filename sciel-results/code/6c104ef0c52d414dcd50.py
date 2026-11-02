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

def _find_rank10_subtensor(time_budget):
    """
    Find a rank-10 decomposition for M_{2,2,3}.
    Dimensions: U (4x10), V (6x10), W (6x10).
    Tensor T (4, 6, 6) where T[i,j,k] = 1 if (i,j,k) corresponds to (a,b,c) in 2x2x3.
    i = a*2 + b
    j = b*3 + c
    k = c*2 + a
    """
    n, m, p = 2, 2, 3
    rank = 10
    nm, mp, pn = n * m, m * p, p * n
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
                
    start_time = time.time()
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple random initializations
    for init_idx in range(5):
        if time.time() - start_time > time_budget:
            break
            
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Scale down to avoid overflow
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        def objective_and_grad(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = M - T_true
            sq_res = np.sum(R**2)
            
            # Gradients
            # grad_U[i,r] = 2 * sum_jk R[i,j,k] V[j,r] W[k,r]
            A = np.einsum('ijk,kr->ijr', R, W)
            grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
            
            # grad_V[j,r] = 2 * sum_ik R[i,j,k] U[i,r] W[k,r]
            B = np.einsum('ijk,kr->ijr', R, W)
            grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
            
            # grad_W[k,r] = 2 * sum_ij R[i,j,k] U[i,r] V[j,r]
            D = np.einsum('ir,jr->ijr', U, V)
            grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return sq_res, grad

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 10000})
            
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
                
        except Exception:
            continue
            
    if best_res < 1e-9:
        return best_U, best_V, best_W, best_res
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try to find Rank 30 via 3x Rank 10 sub-tensors
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        search_budget = min(remaining_time - 10, 60)
        result = _find_rank10_subtensor(search_budget)
        if result is not None:
            U_sub, V_sub, W_sub, score = result
            if score < 1e-9:
                # Construct full Rank 30 matrices
                rank_full = 30
                U_full = np.zeros((n * m, rank_full))
                V_full = np.zeros((m * p, rank_full))
                W_full = np.zeros((p * n, rank_full))
                
                # U is the same for all 3 blocks
                U_full[:, 0:10] = U_sub
                U_full[:, 10:20] = U_sub
                U_full[:, 20:30] = U_sub
                
                # V and W are tiled
                # Block 1: cols 0-2 of B, rows 0-2 of C
                V_full[0:6, 0:10] = V_sub
                W_full[0:6, 0:10] = W_sub
                
                # Block 2: cols 3-5 of B, rows 3-5 of C
                V_full[6:12, 10:20] = V_sub
                W_full[6:12, 10:20] = W_sub
                
                # Block 3: cols 6-8 of B, rows 6-8 of C
                V_full[12:18, 20:30] = V_sub
                W_full[12:18, 20:30] = W_sub
                
                # Verify full solution
                res_full = _compute_residual(U_full, V_full, W_full, n, m, p)
                if res_full < 1e-9:
                    def to_list(arr):
                        return arr.tolist()
                    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
                
    # Fall back to Rank 32
    U32, V32, W32 = _build_rank32()
    def to_list(arr):
        return arr.tolist()
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}
