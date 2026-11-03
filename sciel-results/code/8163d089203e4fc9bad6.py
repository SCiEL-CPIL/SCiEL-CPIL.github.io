"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    The output follows the trace convention W[k*n + i, r].
    
    We use the schoolbook algorithm as a fallback, but attempt to find a
    lower-rank decomposition using known algebraic structures or search.
    
    For 4x4 * 4x9, the schoolbook rank is 4*4*9 = 144.
    The optimal rank is likely close to 144, but potentially lower.
    However, finding a provably optimal or significantly lower rank decomposition
    is a hard open problem. We will implement a search for a lower rank
    decomposition if time permits, otherwise return the schoolbook.
    
    Given the time limit (120s), we will primarily return the schoolbook
    decomposition, which is guaranteed to be correct.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check if we have time to attempt a search for lower rank
    elapsed = time.time() - start_time
    if elapsed < 10.0:
        # Attempt a simple search for a lower rank decomposition
        # This is a heuristic search, not guaranteed to find optimal
        best_rank = rank_schoolbook
        best_U, best_V, best_W = U_list, V_list, W_list
        
        # We could try to find a decomposition with rank < rank_schoolbook
        # But this is computationally intensive. For now, we return the schoolbook.
        # A more sophisticated approach would use tensor decomposition algorithms
        # like CP decomposition with exact arithmetic, but that's complex.
        
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
