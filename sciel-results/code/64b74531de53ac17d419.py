"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A_12 * B_21 (Schoolbook)
    # A_12 is 2x1 (col 2 of A), B_21 is 1x5 (row 2 of B)
    # A indices: i in 0..1, j=2 -> idx i*3+2
    # B indices: j=2, k in 0..4 -> idx 2*5+k
    # W indices: k in 0..4, i in 0..1 -> idx k*2+i
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Block 2: A_11 * B_11a (Strassen)
    # A_11 is 2x2 (cols 0,1), B_11a is 2x2 (rows 0,1, cols 0,1)
    # A indices: i in 0..1, j in 0..1 -> idx i*3+j
    # B indices: j in 0..1, k in 0..1 -> idx j*5+k
    # W indices: k in 0..1, i in 0..1 -> idx k*2+i
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B10-B11) -> C01
    # M4 = A11(B01-B00) -> C10
    # M5 = (A00+A01)B11 -> C01
    # M6 = (A10-A00)(B00+B10) -> C11
    # M7 = (A01-A11)(B11+B01) -> C11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We add contributions to U, V, W for each term
    
    # Term 1
    U[0][col] += 1; U[4][col] += 1 # A00, A11
    V[0][col] += 1; V[6][col] += 1 # B00, B11
    W[0][col] += 1; W[3][col] += 1 # C00, C11
    col += 1
    
    # Term 2
    U[3][col] += 1; U[4][col] += 1 # A10, A11
    V[0][col] += 1 # B00
    W[1][col] += 1 # C10
    col += 1
    
    # Term 3
    U[0][col] += 1 # A00
    V[5][col] += 1; V[6][col] += -1 # B10, B11
    W[2][col] += 1 # C01
    col += 1
    
    # Term 4
    U[4][col] += 1 # A11
    V[1][col] += 1; V[0][col] += -1 # B01, B00
    W[1][col] += 1 # C10
    col += 1
    
    # Term 5
    U[0][col] += 1; U[1][col] += 1 # A00, A01
    V[6][col] += 1 # B11
    W[2][col] += 1 # C01
    col += 1
    
    # Term 6
    U[3][col] += 1; U[0][col] += -1 # A10, A00
    V[0][col] += 1; V[5][col] += 1 # B00, B10
    W[3][col] += 1 # C11
    col += 1
    
    # Term 7
    U[1][col] += 1; U[4][col] += -1 # A01, A11
    V[6][col] += 1; V[1][col] += 1 # B11, B01
    W[3][col] += 1 # C11
    col += 1
    
    # Block 3: A_11 * B_11b (Strassen)
    # B_11b is 2x2 (rows 0,1, cols 2,3)
    # B indices: j in 0..1, k in 2..3 -> idx j*5+k
    # B02 -> 2, B03 -> 3, B12 -> 7, B13 -> 8
    # W indices: k in 2..3, i in 0..1 -> idx k*2+i
    # C02 -> 4, C03 -> 6, C12 -> 5, C13 -> 7
    
    # Term 1
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1
    col += 1
    
    # Term 2
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1
    col += 1
    
    # Term 3
    U[0][col] += 1
    V[7][col] += 1; V[8][col] += -1
    W[6][col] += 1
    col += 1
    
    # Term 4
    U[4][col] += 1
    V[3][col] += 1; V[2][col] += -1
    W[5][col] += 1
    col += 1
    
    # Term 5
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[6][col] += 1
    col += 1
    
    # Term 6
    U[3][col] += 1; U[0][col] += -1
    V[2][col] += 1; V[7][col] += 1
    W[7][col] += 1
    col += 1
    
    # Term 7
    U[1][col] += 1; U[4][col] += -1
    V[8][col] += 1; V[3][col] += 1
    W[7][col] += 1
    col += 1
    
    # Block 4: A_11 * B_11c (Schoolbook)
    # B_11c is 2x1 (rows 0,1, col 4)
    # B indices: j in 0..1, k=4 -> idx j*5+4
    # B04 -> 4, B14 -> 9
    # W indices: k=4, i in 0..1 -> idx 4*2+i
    # C04 -> 8, C14 -> 9
    
    for i in range(2):
        for j in range(2):
            U[i * 3 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
