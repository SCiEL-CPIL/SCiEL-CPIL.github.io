"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    
    # Attempt to find a lower rank decomposition using a simple search or known results.
    # For 2x3x4, the rank is known to be 20.
    # If we can find a valid rank 20 decomposition, we return it.
    # Otherwise, we return the schoolbook one.
    
    # Since I cannot reliably generate the rank 20 terms on the fly without a pre-computed table,
    # and hardcoding a potentially incorrect table is risky, 
    # I will return the schoolbook decomposition which is guaranteed to be correct.
    # A future evolution might find the rank 20 terms.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
