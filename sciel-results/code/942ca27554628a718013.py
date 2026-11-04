"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using numpy einsum.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """
    Construct the schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # 1. Generate Schoolbook Decomposition (Rank 125 for 5x5x5)
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify schoolbook (should be exact)
    if not _verify_fast(U, V, W, n, m, p):
        pass

    # 2. Quick attempt: try rank 124 with random perturbation search
    # This is extremely unlikely to succeed for 5x5x5, but we try briefly.
    if time.time() < deadline - 20:
        target_rank = n * m * p - 1
        
        np.random.seed(42)
        U_gd = U[:, :target_rank].copy()
        V_gd = V[:, :target_rank].copy()
        W_gd = W[:, :target_rank].copy()
        
        # Small perturbation to break degeneracy
        U_gd += np.random.randn(n*m, target_rank) * 0.01
        V_gd += np.random.randn(m*p, target_rank) * 0.01
        W_gd += np.random.randn(p*n, target_rank) * 0.01
        
        def compute_error(U, V, W):
            U_r = U.reshape(n, m, -1)
            V_r = V.reshape(m, p, -1)
            W_r = W.reshape(p, n, -1)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
            return np.sum((T - 1.0)**2)
        
        err = compute_error(U_gd, V_gd, W_gd)
        
        # Random perturbation search (very fast, limited steps)
        best_err = err
        best_U, best_V, best_W = U_gd.copy(), V_gd.copy(), W_gd.copy()
        
        for step in range(30):
            if time.time() > deadline - 5:
                break
            
            scale = 0.05 * (1 - step/30)
            U_try = best_U + np.random.randn(n*m, target_rank) * scale
            V_try = best_V + np.random.randn(m*p, target_rank) * scale
            W_try = best_W + np.random.randn(p*n, target_rank) * scale
            
            try:
                err_try = compute_error(U_try, V_try, W_try)
                if err_try < best_err:
                    best_err = err_try
                    best_U, best_V, best_W = U_try, V_try, W_try
            except:
                pass
        
        # Check if we found a valid solution
        if best_err < 1e-6:
            if _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
