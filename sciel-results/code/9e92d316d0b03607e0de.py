"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # Based on the induction argument, rank might be around 2p+2 = 32
    # Or even lower. Let's try to construct rank 32 explicitly.
    
    def try_construct_rank_optimal():
        # Base case k=0: 4 terms
        # Pairs k=1,2; 3,4; ...; 13,14: 7 pairs, 2 terms each = 14 terms
        # Total: 18 terms
        R = 4 + 2 * 7  # 18
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Base case k=0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        current_rank = 4
        for pair_idx in range(7):
            k1 = 2 * pair_idx + 1
            k2 = 2 * pair_idx + 2
            
            c1 = current_rank
            c2 = current_rank + 1
            
            U[0, c1] = 1
            U[1, c2] = 1
            U[2, c1] = 1
            U[3, c2] = 1
            
            V[0 * p + k1, c1] = 1
            V[0 * p + k2, c1] = 1
            V[1 * p + k1, c2] = 1
            V[1 * p + k2, c2] = 1
            
            W[k1 * n + 0, c1] = 1
            W[k2 * n + 0, c1] = 1
            W[k1 * n + 0, c2] = 1
            W[k2 * n + 0, c2] = 1
            W[k1 * n + 1, c1] = 1
            W[k2 * n + 1, c1] = 1
            W[k1 * n + 1, c2] = 1
            W[k2 * n + 1, c2] = 1
            
            current_rank += 2
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return None
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result_opt = try_construct_rank_optimal()
    if result_opt:
        return result_opt
        
    return best_result
# EVOLVE-BLOCK-END
