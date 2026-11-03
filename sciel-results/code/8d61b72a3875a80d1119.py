"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i*m + j. i in 0..1, j in 0..1
    # B indices: j*p + k. j in 0..1, k in 0..1
    # C indices: k*n + i. k in 0..1, i in 0..1
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][col] = 1; U[3][col] = 1  # A00, A11 (0*3+0, 1*3+1)
    V[0][col] = 1; V[5][col] = 1  # B00, B11 (0*5+0, 1*5+1)
    # C00 += M1, C11 += M1
    W[0][col] = 1  # C00 (0*2+0)
    W[1][col] = 1  # C11 (1*2+1)
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[3][col] = 1; U[4][col] = 1  # A10, A11
    V[0][col] = 1                 # B00
    # C10 += M2
    W[2][col] = 1  # C10 (0*2+1)
    col += 1
    
    # M3 = A00 * (B01 - B11)
    U[0][col] = 1                 # A00
    V[1][col] = 1; V[6][col] = -1 # B01, B11
    # C01 += M3
    W[3][col] = 1  # C01 (1*2+0)
    col += 1
    
    # M4 = A11 * (B10 - B00)
    U[4][col] = 1                 # A11
    V[5][col] = 1; V[0][col] = -1 # B10, B00
    # C00 += M4, C10 += M4
    W[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[0][col] = 1; U[1][col] = 1  # A00, A01
    V[6][col] = 1                 # B11
    # C00 -= M5, C01 += M5
    W[0][col] = -1
    W[3][col] = 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3][col] = 1; U[0][col] = -1 # A10, A00
    V[0][col] = 1; V[1][col] = 1  # B00, B01
    # C11 += M6
    W[1][col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][col] = 1; U[4][col] = -1 # A01, A11
    V[5][col] = 1; V[6][col] = 1  # B10, B11
    # C00 += M7, C11 -= M7
    W[0][col] = 1
    W[1][col] = -1
    col += 1
    
    # 2. Schoolbook for A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # i in 0..1, j in 0..1, k in 2..4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * 3 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
                
    # 3. Schoolbook for A[0:2, 2] * B[2, 0:5] -> C[0:2, 0:5]
    # i in 0..1, j=2, k in 0..4
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
