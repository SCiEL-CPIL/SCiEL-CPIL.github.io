"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition for 2x3x7
    # Split m=3 into m1=1, m2=2
    # m1=1 part: 2x1x7, rank 14
    # m2=2 part: 2x2x7, split p=7 into 2+2+2+1
    #   3 blocks of 2x2x2 (rank 7 each)
    #   1 block of 2x2x1 (rank 4)
    # Total rank: 14 + 3*7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0 (m=1 part)
    # C_ik += A_i0 * B_0k
    for k in range(p):
        # Term for i=0
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        # Term for i=1
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: j=1,2 (m=2 part)
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    def add_strassen_block(j0, j1, k0, k1):
        nonlocal col
        # A indices
        A00 = 0 * m + j0
        A01 = 0 * m + j1
        A10 = 1 * m + j0
        A11 = 1 * m + j1
        
        # B indices
        B00 = j0 * p + k0
        B01 = j0 * p + k1
        B10 = j1 * p + k0
        B11 = j1 * p + k1
        
        # C indices
        C00 = k0 * n + 0
        C01 = k1 * n + 0
        C10 = k0 * n + 1
        C11 = k1 * n + 1
        
        # M1
        U[A00][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M2
        U[A10][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        col += 1
        
        # M3
        U[A00][col] += 1
        V[B01][col] += 1
        V[B11][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M4
        U[A11][col] += 1
        V[B10][col] += 1
        V[B00][col] += -1
        W[C00][col] += -1
        W[C10][col] += 1
        col += 1
        
        # M5
        U[A00][col] += 1
        U[A01][col] += 1
        V[B11][col] += 1
        W[C01][col] += 1
        col += 1
        
        # M6
        U[A10][col] += 1
        U[A00][col] += -1
        V[B00][col] += 1
        V[B01][col] += 1
        W[C00][col] += 1
        col += 1
        
        # M7
        U[A01][col] += 1
        U[A11][col] += -1
        V[B10][col] += 1
        V[B11][col] += 1
        W[C11][col] += 1
        col += 1

    # Block 0: k=1,2
    add_strassen_block(1, 2, 1, 2)
    # Block 1: k=3,4
    add_strassen_block(1, 2, 3, 4)
    # Block 2: k=5,6
    add_strassen_block(1, 2, 5, 6)
    
    # Part 3: j=1,2, k=0 (2x2x1 part)
    # C00 += A01*B10 + A02*B20
    # C10 += A11*B10 + A12*B20
    # A indices
    A01 = 0 * m + 1
    A02 = 0 * m + 2
    A11 = 1 * m + 1
    A12 = 1 * m + 2
    
    # B indices
    B10 = 1 * p + 0
    B20 = 2 * p + 0
    
    # C indices
    C00 = 0 * n + 0
    C10 = 0 * n + 1
    
    # Term 1: A01 * B10
    U[A01][col] = 1
    V[B10][col] = 1
    W[C00][col] = 1
    col += 1
    
    # Term 2: A02 * B20
    U[A02][col] = 1
    V[B20][col] = 1
    W[C00][col] = 1
    col += 1
    
    # Term 3: A11 * B10
    U[A11][col] = 1
    V[B10][col] = 1
    W[C10][col] = 1
    col += 1
    
    # Term 4: A12 * B20
    U[A12][col] = 1
    V[B20][col] = 1
    W[C10][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
