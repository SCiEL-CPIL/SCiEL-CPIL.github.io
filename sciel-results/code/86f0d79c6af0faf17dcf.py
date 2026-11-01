"""Matrix multiplication 2x3x9 - improved decomposition of rank 41."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # U shape (n*m, rank) = (6, rank)
    # V shape (m*p, rank) = (27, rank)
    # W shape (p*n, rank) = (18, rank)
    
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x9)
    # Split B[0:2, :] into 4 blocks of 2 columns and 1 block of 1 column
    # B cols: [0,1], [2,3], [4,5], [6,7], [8]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A: A00->0, A01->1, A10->3, A11->4 (since m=3, A[i,j] index is i*3+j)
    # A00: 0*3+0 = 0
    # A01: 0*3+1 = 1
    # A10: 1*3+0 = 3
    # A11: 1*3+1 = 4
    
    # Indices for B: B[j,k] index is j*9+k
    # Block 0 (cols 0,1): B00=0, B01=1, B10=9, B11=10
    # Block 1 (cols 2,3): B00=2, B01=3, B10=11, B11=12
    # Block 2 (cols 4,5): B00=4, B01=5, B10=13, B11=14
    # Block 3 (cols 6,7): B00=6, B01=7, B10=15, B11=16
    # Block 4 (col 8):    B00=8, B10=17
    
    # Indices for W: C[i,k] index is k*n+i = k*2+i
    # Block 0 (cols 0,1): C00=0, C01=1, C10=2, C11=3
    # Block 1 (cols 2,3): C00=4, C01=5, C10=6, C11=7
    # Block 2 (cols 4,5): C00=8, C01=9, C10=10, C11=11
    # Block 3 (cols 6,7): C00=12, C01=13, C10=14, C11=15
    # Block 4 (col 8):    C00=16, C10=17
    
    def add_strassen(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11, start_idx):
        # M1
        U[A00][start_idx] += 1
        U[A11][start_idx] += 1
        V[B00][start_idx] += 1
        V[B11][start_idx] += 1
        W[C00][start_idx] += 1
        W[C11][start_idx] += 1
        start_idx += 1
        # M2
        U[A10][start_idx] += 1
        U[A11][start_idx] += 1
        V[B00][start_idx] += 1
        W[C10][start_idx] += 1
        W[C11][start_idx] += -1
        start_idx += 1
        # M3
        U[A00][start_idx] += 1
        V[B01][start_idx] += 1
        V[B11][start_idx] += -1
        W[C01][start_idx] += 1
        W[C11][start_idx] += 1
        start_idx += 1
        # M4
        U[A11][start_idx] += 1
        V[B10][start_idx] += 1
        V[B00][start_idx] += -1
        W[C00][start_idx] += 1
        W[C10][start_idx] += 1
        start_idx += 1
        # M5
        U[A00][start_idx] += 1
        U[A01][start_idx] += 1
        V[B11][start_idx] += 1
        W[C00][start_idx] += -1
        W[C01][start_idx] += 1
        start_idx += 1
        # M6
        U[A10][start_idx] += 1
        U[A00][start_idx] += -1
        V[B00][start_idx] += 1
        V[B01][start_idx] += 1
        W[C11][start_idx] += 1
        start_idx += 1
        # M7
        U[A01][start_idx] += 1
        U[A11][start_idx] += -1
        V[B10][start_idx] += 1
        V[B11][start_idx] += 1
        W[C00][start_idx] += 1
        start_idx += 1
        
        return start_idx

    # Block 0: Cols 0,1
    idx = add_strassen(0, 1, 3, 4, 0, 1, 9, 10, 0, 1, 2, 3, idx)
    # Block 1: Cols 2,3
    idx = add_strassen(0, 1, 3, 4, 2, 3, 11, 12, 4, 5, 6, 7, idx)
    # Block 2: Cols 4,5
    idx = add_strassen(0, 1, 3, 4, 4, 5, 13, 14, 8, 9, 10, 11, idx)
    # Block 3: Cols 6,7
    idx = add_strassen(0, 1, 3, 4, 6, 7, 15, 16, 12, 13, 14, 15, idx)
    
    # Part 1b: A[:, 0:2] * B[0:2, 8] (2x2 * 2x1)
    # Schoolbook: 4 mults
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # A00=0, A01=1, A10=3, A11=4
    # B00=8, B10=17
    # C00=16, C10=17
    
    # M1: A00 * B00
    U[0][idx] = 1
    V[8][idx] = 1
    W[16][idx] = 1
    idx += 1
    # M2: A01 * B10
    U[1][idx] = 1
    V[17][idx] = 1
    W[16][idx] = 1
    idx += 1
    # M3: A10 * B00
    U[3][idx] = 1
    V[8][idx] = 1
    W[17][idx] = 1
    idx += 1
    # M4: A11 * B10
    U[4][idx] = 1
    V[17][idx] = 1
    W[17][idx] = 1
    idx += 1
    
    # Part 2: A[:, 2:3] * B[2:3, :] (2x1 * 1x9)
    # A02=2, A12=5
    # B2k = 18+k
    # C0k = 2k, C1k = 2k+1
    for k in range(9):
        # C0k += A02 * B2k
        U[2][idx] = 1
        V[18 + k][idx] = 1
        W[2 * k][idx] = 1
        idx += 1
        # C1k += A12 * B2k
        U[5][idx] = 1
        V[18 + k][idx] = 1
        W[2 * k + 1][idx] = 1
        idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
