import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 120
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Verification function using einsum
    def verify(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9)
    
    # Try structured searches for lower rank
    # For each target rank, try random initializations with half-integer entries
    target_ranks = [119, 118, 117, 116, 115, 114, 113, 112, 111, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(seed * 7919 + target_rank)
            
            # Generate random half-integer entries in range [-2, 2]
            # Half-integers: ..., -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
            def random_half_integers(shape):
                # Generate integers from -4 to 4, then divide by 2
                ints = np.random.randint(-4, 5, size=shape)
                return ints.astype(np.float64) / 2.0
            
            U = random_half_integers((n * m, target_rank))
            V = random_half_integers((m * p, target_rank))
            W = random_half_integers((p * n, target_rank))
            
            if verify(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
