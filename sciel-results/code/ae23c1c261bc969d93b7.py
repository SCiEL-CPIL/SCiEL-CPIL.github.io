"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * p for n=2, m>=2
    rank = n * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill matrices based on the decomposition:
    # r corresponds to (i, k) pair
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1 if r corresponds to (i, k)
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            # Set W
            W[k * n + i][r] = 1
            
            # Set U and V for all j
            for j in range(m):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
