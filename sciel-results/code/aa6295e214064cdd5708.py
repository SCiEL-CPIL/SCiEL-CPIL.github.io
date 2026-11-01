import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 2x2x2 Strassen
    # A indices: 0,1 (row 0), 3,4 (row 1)
    # B indices: 0,1 (col 0), 3,4 (col 1)
    # C indices: 0,1 (col 0), 2,3 (col 1)
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A11(B00-B10)
    U[4][col] = 1
    V[0][col] = 1; V[3][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A[2,5] * B[9,10] -> C[0,1,2,3]
    # A[2]B[9] -> C[0]
    U[2][col] = 1; V[9][col] = 1; W[0][col] = 1
    col += 1
    # A[2]B[10] -> C[1]
    U[2][col] = 1; V[10][col] = 1; W[1][col] = 1
    col += 1
    # A[5]B[9] -> C[2]
    U[5][col] = 1; V[9][col] = 1; W[2][col] = 1
    col += 1
    # A[5]B[10] -> C[3]
    U[5][col] = 1; V[10][col] = 1; W[3][col] = 1
    col += 1
    
    # Block 3: A[0,1,3,4] * B[2,5] -> C[4,5]
    # A[0]B[2] -> C[4]
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1
    col += 1
    # A[1]B[5] -> C[4]
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1
    col += 1
    # A[3]B[2] -> C[5]
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1
    col += 1
    # A[4]B[5] -> C[5]
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1
    col += 1
    
    # Block 4: A[2,5] * B[8] -> C[4,5]
    # A[2]B[8] -> C[4]
    U[2][col] = 1; V[8][col] = 1; W[4][col] = 1
    col += 1
    # A[5]B[8] -> C[5]
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
