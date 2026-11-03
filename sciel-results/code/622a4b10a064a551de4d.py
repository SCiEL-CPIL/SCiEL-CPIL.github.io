"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    rank = 18
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for j in range(m):
        for k in range(p):
            # For each (j, k), we create one term that covers i=0 and i=1
            # U indices: i*m + j
            # V indices: j*p + k
            # W indices: k*n + i
            
            # Set U for i=0 and i=1
            U[0 * m + j][column] = 1
            U[1 * m + j][column] = 1
            
            # Set V for j, k
            V[j * p + k][column] = 1
            
            # Set W for k, i=0 and k, i=1
            W[k * n + 0][column] = 1
            W[k * n + 1][column] = 1
            
            column += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
