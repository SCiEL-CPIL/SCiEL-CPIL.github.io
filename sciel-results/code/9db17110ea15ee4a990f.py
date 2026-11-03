"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, i*n+k] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_school = n*m*p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[i*n+k, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def cp_als(T, rank, max_iters=20, seed=42):
        np.random.seed(seed)
        nm, mp, pn = T.shape
        U = np.random.randn(nm, rank) * 0.3
        V = np.random.randn(mp, rank) * 0.3
        W = np.random.randn(pn, rank) * 0.3
        
        for it in range(max_iters):
            for r in range(rank):
                U[:, r] = np.tensordot(T, np.outer(V[:, r], W[:, r]), axes=([1, 2], [0, 1]))
            for r in range(rank):
                V[:, r] = np.tensordot(T, np.outer(U[:, r], W[:, r]), axes=([0, 2], [0, 1]))
            for r in range(rank):
                W[:, r] = np.tensordot(T, np.outer(U[:, r], V[:, r]), axes=([0, 1], [0, 1]))
        
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(recon - T))
        return U, V, W, err
    
    # Try ranks in increasing order
    for rank in range(30, 101, 2):
        if time.time() > deadline:
            break
        
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline:
                break
            
            U, V, W, err = cp_als(T, rank, max_iters=15, seed=seed)
            
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify(U_r, V_r, W_r):
                    best_rank = rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
                elif verify(U, V, W):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
