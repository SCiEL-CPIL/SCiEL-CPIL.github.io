"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication.
    For 3x4 by 4x7, the schoolbook rank is 84.
    We return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = 150.0 # Evaluator kills after 150s, we keep margin.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Using lists of lists for compatibility with the seed interface
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
                
    # Time check (should be instantaneous)
    if time.time() > deadline - 10:
        # Return best found (which is the schoolbook)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
