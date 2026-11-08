import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # The tensor M is defined by M[(i*m+j), (j*p+k), (k*n+i)] = 1
    M = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i * m + j, j * p + k, k * n + i] = 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    start_time = time.time()
    time_limit = 110
    
    # Try to find a lower rank decomposition
    for target_rank in range(44, 19, -1):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(42 + target_rank)
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        err = np.inf
        for it in range(100):
            # Update U
            Z = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, target_rank)
            M_flat = M.reshape(n * m, -1)
            try:
                U = np.linalg.lstsq(Z.T, M_flat.T, rcond=None)[0].T
            except:
                pass
                
            # Update V
            M_flat_V = M.reshape(m * p, -1)
            Z_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, target_rank)
            try:
                V = np.linalg.lstsq(Z_V.T, M_flat_V.T, rcond=None)[0].T
            except:
                pass
                
            # Update W
            M_flat_W = M.reshape(p * n, -1)
            Z_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, target_rank)
            try:
                W = np.linalg.lstsq(Z_W.T, M_flat_W.T, rcond=None)[0].T
            except:
                pass
            
            M_approx = (U[:, None, None] * V[None, :, None] * W[None, None, :]).sum(axis=2)
            err = np.linalg.norm(M - M_approx)
            if err < 1e-9:
                break
        
        if err < 1e-7:
            # Try rounding
            for rounding_factor in [1, 2]:
                U_round = np.round(U * rounding_factor) / rounding_factor
                V_round = np.round(V * rounding_factor) / rounding_factor
                W_round = np.round(W * rounding_factor) / rounding_factor
                
                M_approx_round = (U_round[:, None, None] * V_round[None, :, None] * W_round[None, None, :]).sum(axis=2)
                if np.allclose(M, M_approx_round, atol=1e-9):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    break
            if best_rank == target_rank:
                break

    # Final verification
    M_check = (best_U[:, None, None] * best_V[None, :, None] * best_W[None, None, :]).sum(axis=2)
    if not np.allclose(M, M_check, atol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
