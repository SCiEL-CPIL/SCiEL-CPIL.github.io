"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition for 2x2x5 matrix multiplication: rank 20
    # Each term corresponds to one scalar multiplication A[i,j]*B[j,k] contributing to C[i,k]
    total_terms = n * m * p
    U = [[0] * total_terms for _ in range(n * m)]
    V = [[0] * total_terms for _ in range(m * p)]
    W = [[0] * total_terms for _ in range(p * n)]
    
    # Assign each (i,j,k) triple to a unique rank index
    counter = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                U[row_u][counter] = 1
                V[row_v][counter] = 1
                W[row_w][counter] = 1
                counter += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
