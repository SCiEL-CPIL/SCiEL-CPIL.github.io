"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """Construct the rank-minimal (or best found) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    Returns dict with 'U', 'V', 'W' factor matrices of shapes:
      U: (n*m, rank) = (15, rank)
      V: (m*p, rank) = (25, rank)  
      W: (p*n, rank) = (15, rank)
    
    The decomposition satisfies:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    for all valid indices, following the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists (or use numpy for speed)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
