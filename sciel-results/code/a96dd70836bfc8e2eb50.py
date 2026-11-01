"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's decomposition of rank 7 for 2x2 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k] (row index k*n+i)
    
    # Product 1: (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Product 3: (A00 - A01) * (B01 - B11)
    U[0][2] = 1; U[1][2] = -1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Product 4: A11 * (B10 - B11)
    U[3][3] = 1
    V[2][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Product 5: A00 * (B00 - B10)
    U[0][4] = 1
    V[0][4] = 1; V[2][4] = -1
    W[0][4] = -1; W[1][4] = 1
    
    # Product 6: (A01 + A11) * B11
    U[1][5] = 1; U[3][5] = 1
    V[3][5] = 1
    W[3][5] = 1
    
    # Product 7: (A00 + A10) * B01
    U[0][6] = 1; U[2][6] = 1
    V[1][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
