"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def try_rank(rank, max_iters=300, n_seeds=3):
        for seed in range(n_seeds):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(seed * 12345 + 67890)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        b = np.ones(p)
                        U[i * m + j, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        b = np.ones(n)
                        V[j * p + k, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        b = np.ones(m)
                        W[k * n + i, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                
                if verify(U, V, W):
                    return U, V, W, True
        
        return None, None, None, False
    
    # Try lower ranks
    for rank in range(17, 10, -1):
        if time.time() > deadline - 15:
            break
        U, V, W, ok = try_rank(rank)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
