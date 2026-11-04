"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-33 decomposition: 3 Strassen 2x2x2 blocks + schoolbook for column 2
    # A is 2x3, B is 3x6, C = A*B is 2x6
    # U: 6 x 33, V: 18 x 33, W: 12 x 33
    # U index: i*m+j = i*3+j (i=0..1, j=0..2)
    # V index: j*p+k = j*6+k (j=0..2, k=0..5)
    # W index: k*n+i = k*2+i (k=0..5, i=0..1)
    
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each of the 3 Strassen blocks (columns 2k and 2k+1 of B)
    # Strassen uses 7 multiplications for 2x2 matrix multiplication
    # Block k: A[:, 0:2] * B[0:2, 2k:2k+2]
    # Strassen formulas for C = A2 * B2 where A2, B2 are 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for k in range(3):
        base = 7 * k
        # Strassen block k multiplies A[:, 0:2] by B[0:2, 2k:2k+2]
        # A indices: A[i, 0] = U[i*3+0], A[i, 1] = U[i*3+1]
        # B indices: B[j, 2k] = V[j*6+2k], B[j, 2k+1] = V[j*6+2k+1]
        # C indices: C[i, 2k] = W[(2k)*2+i], C[i, 2k+1] = W[(2k+1)*2+i]
        
        # M1 = (A00+A11)*(B00+B11)
        c = base + 0
        U[0][c] = 1  # A00
        U[5][c] = 1  # A11
        V[0][c] = 1  # B00 = B[0,2k]
        V[6][c] = 1  # B11 = B[1,2k+1]
        W[4*k][c] = 1      # C00 = C[0,2k]
        W[4*k+3][c] = 1    # C11 = C[1,2k+1]
        
        # M2 = (A10+A11)*B00
        c = base + 1
        U[3][c] = 1  # A10
        U[5][c] = 1  # A11
        V[0][c] = 1  # B00
        W[4*k+1][c] = 1   # C10 = C[1,2k]
        W[4*k+3][c] = -1  # C11
        
        # M3 = A00*(B10-B11)
        c = base + 2
        U[0][c] = 1  # A00
        V[6][c] = 1  # B10 = B[1,2k]
        V[7][c] = -1 # B11 = B[1,2k+1]
        W[4*k+2][c] = 1   # C01 = C[0,2k+1]
        W[4*k+3][c] = 1   # C11
        
        # M4 = A11*(B01-B00)
        c = base + 3
        U[5][c] = 1  # A11
        V[1][c] = 1  # B01 = B[0,2k+1]
        V[0][c] = -1 # B00 = B[0,2k]
        W[4*k][c] = 1     # C00
        W[4*k+1][c] = 1   # C10
        
        # M5 = (A00+A01)*B11
        c = base + 4
        U[0][c] = 1  # A00
        U[1][c] = 1  # A01
        V[7][c] = 1  # B11 = B[1,2k+1]
        W[4*k][c] = -1    # C00
        W[4*k+2][c] = 1   # C01
        
        # M6 = (A10-A00)*(B00+B10)
        c = base + 5
        U[3][c] = 1  # A10
        U[0][c] = -1 # A00
        V[0][c] = 1  # B00
        V[6][c] = 1  # B10
        W[4*k+3][c] = 1   # C11
        
        # M7 = (A01-A11)*(B01+B11)
        c = base + 6
        U[1][c] = 1  # A01
        U[5][c] = -1 # A11
        V[1][c] = 1  # B01
        V[7][c] = 1  # B11
        W[4*k][c] = 1     # C00
    
    # Extra terms for column 2 of A (A[:, 2]) times B[2, :]
    # A[i, 2] = U[i*3+2], B[2, k] = V[2*6+k] = V[12+k]
    # C[i, k] = W[k*2+i]
    for j in range(6):
        # A[0,2] * B[2,j] -> C[0,j]
        c = 21 + j
        U[2][c] = 1   # A[0,2]
        V[12+j][c] = 1  # B[2,j]
        W[j*2][c] = 1   # C[0,j]
        
        # A[1,2] * B[2,j] -> C[1,j]
        c = 27 + j
        U[5][c] = 1   # A[1,2]
        V[12+j][c] = 1  # B[2,j]
        W[j*2+1][c] = 1   # C[1,j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

