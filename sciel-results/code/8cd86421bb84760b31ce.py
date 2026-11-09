"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 45
    # We will try to find a lower rank decomposition using Strassen for the 3x3x3 parts
    
    # Strategy: The 3x3x5 multiplication can be viewed as 3 independent 3x3x3 multiplications
    # if we partition the 5 columns into 3+2, but that doesn't help directly.
    # However, we can try to find a decomposition with rank < 45.
    
    # Let's try to construct a rank 27 decomposition using Strassen for 3x3x3?
    # Strassen is for 2x2x2. For 3x3x3, the minimal rank is known to be 19? No, 3x3x3 is 19?
    # Actually, for 3x3x3, the lower bound is 19, upper bound 23.
    # If we use 3 blocks of 3x3x3, we need rank 3 * rank(3x3x3).
    # If rank(3x3x3) is 19, total rank 57 > 45.
    # So simple block decomposition doesn't help.
    
    # Let's try a numerical optimization to find a lower rank.
    # We will use the tensor network decomposition approach.
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds budget
    
    # We will try to find a decomposition with rank r
    # We start with r = 45 and try to reduce it.
    
    # Let's try a random initialization and gradient descent on the rank-27 decomposition?
    # Or use the fact that the tensor is sparse.
    
    # Actually, let's just return the schoolbook solution for now, as finding a better one
    # requires significant computational effort and is not guaranteed.
    
    # But we can try a simple heuristic:
    # The tensor T_{i,j,k} = e_{i,j} * e_{j,k}
    # We want to find U, V, W such that sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,j} delta_{j,k}
    
    # Let's try rank 36 (9 * 4)?
    # Or rank 40?
    
    # For now, let's stick to the schoolbook rank 45 to ensure validity.
    # If we have time, we can try to optimize.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
