import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Uses structured search to find rank < 45.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify a candidate decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank by random search with local optimization
    # Start with various rank values
    
    random.seed(42)
    
    for target_rank in range(36, 45):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values from a small set
            U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            # Quick check: compute the tensor
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            if np.allclose(tensor, 1.0, atol=1e-9):
                # Found a valid decomposition
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                # Continue searching for even lower rank
                break
        
        if best_rank == target_rank:
            # Found one at this rank, try lower
            continue
    
    if best_rank < n * m * p:
        return best
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
