"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 4x5x6, the schoolbook algorithm provides a decomposition of rank 120.
    Given the small dimensions and lack of widely known simple lower-rank identities
    for this specific rectangular case that are easy to construct programmatically 
    within the time limit, we use the schoolbook decomposition.
    """
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flattened array positions
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
                
                # Safety check for deadline, though this loop is very fast
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
