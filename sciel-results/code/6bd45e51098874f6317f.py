import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
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
    add_trivial(6)
    add_strassen(7, 8)
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    # Tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Expected: T[i,j,k] = 1 if i//m * p + j == k (for A[i//m, i%m] * B[i%m, j] -> C[i//m, j])
    # Actually: C = A @ B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i, j, k] = delta_{i//?}... 
    # The tensor for matrix multiplication: T[i, j, k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # T[i, j, k] = 1 if A_row = i // m, A_col = i % m, B_row = j // p, B_col = j % p, C_row = k // n, C_col = k % n
    # and A_col == B_row and A_row == C_row and B_col == C_col
    
    residual = 0.0
    for i in range(n * m):
        ar, ac = i // m, i % m
        for j in range(m * p):
            br, bc = j // p, j % p
            if ac != br:
                continue
            for k in range(p * n):
                cr, cc = k // n, k % n
                expected = 1.0 if (ar == cr and bc == cc) else 0.0
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i, r] * V[j, r] * W[k, r]
                residual += (actual - expected) ** 2
    return residual

def _try_rank31_search(U32, V32, W32, n, m, p, deadline):
    """Try to find a rank-31 decomposition by perturbing the rank-32 solution."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    rank31 = 31
    # Start from rank-32, zero out one column and optimize
    best_result = None
    best_residual = float('inf')
    
    for drop_col in range(32):
        if time.time() > deadline - 5:
            break
        
        U0 = np.delete(U32, drop_col, axis=1).astype(float)
        V0 = np.delete(V32, drop_col, axis=1).astype(float)
        W0 = np.delete(W32, drop_col, axis=1).astype(float)
        
        # Flatten
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        def objective(x):
            U = x[:4*rank31].reshape(4, rank31)
            V = x[4*rank31:4*rank31+18*rank31].reshape(18, rank31)
            W = x[4*rank31+18*rank31:].reshape(18, rank31)
            # Compute residual efficiently
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Use einsum
            # residual = ||T - T_expected||^2
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            # Build expected tensor
            T_exp = np.zeros((4, 18, 18))
            for i in range(4):
                ar, ac = i // 2, i % 2
                for j in range(18):
                    br, bc = j // 9, j % 9
                    if ac != br:
                        continue
                    for k in range(18):
                        cr, cc = k // 2, k % 2
                        if ar == cr and bc == cc:
                            T_exp[i, j, k] = 1.0
            return np.sum((T - T_exp) ** 2)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_residual:
                best_residual = result.fun
                x_best = result.x
                best_result = (drop_col, x_best)
        except Exception:
            continue
    
    if best_result is not None and best_residual < 1e-8:
        drop_col, x = best_result
        U = x[:4*rank31].reshape(4, rank31).tolist()
        V = x[4*rank31:4*rank31+18*rank31].reshape(18, rank31).tolist()
        W = x[4*rank31+18*rank31:].reshape(18, rank31).tolist()
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    # Try to find rank-31
    result31 = _try_rank31_search(U32, V32, W32, n, m, p, deadline)
    
    if result31 is not None:
        return result31
    
    # Fallback to rank-32
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
# EVOLVE-BLOCK-END
