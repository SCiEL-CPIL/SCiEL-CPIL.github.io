"""Rewritten program using Strassen block decomposition for rank minimization."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will decompose p=15 into 7 blocks of width 2 and 1 block of width 1.
    # For each 2x2 block, we use Strassen (7 terms).
    # For the 2x1 block, we use 4 terms.
    # Total rank = 7*7 + 4 = 53.
    
    rank = 7 * 7 + 4  # 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    
    # Process 7 blocks of width 2 (columns 0-1, 2-3, ..., 12-13)
    for t in range(7):
        # B column indices for this block
        b0 = 2 * t      # B[0, 2t]
        b1 = 2 * t + 1  # B[0, 2t+1]
        b2 = 15 + 2 * t      # B[1, 2t]
        b3 = 15 + 2 * t + 1  # B[1, 2t+1]
        
        # C row indices for this block (output C[i, k] where k is column)
        # C[k*n + i]
        c0 = 2 * t * n      # C[0, 2t] -> index 2t*2 + 0 = 4t
        c1 = 2 * t * n + 1  # C[0, 2t+1] -> index 2t*2 + 1 = 4t+1
        c2 = (2 * t + 1) * n      # C[1, 2t] -> index (2t+1)*2 + 0 = 4t+2
        c3 = (2 * t + 1) * n + 1  # C[1, 2t+1] -> index (2t+1)*2 + 1 = 4t+3
        
        # Strassen terms for 2x2 * 2x2:
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B01 - B11)
        # P4 = A11(B10 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A11 - A01)(B10 + B11)
        #
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # A indices: A00=0, A01=1, A10=2, A11=3 (flattened i*m+j)
        
        # Term 1: P1
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[b0][col] = 1  # B00
        V[b3][col] = 1  # B11
        W[c0][col] = 1  # C00
        W[c3][col] = 1  # C11
        col += 1
        
        # Term 2: P2
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[b0][col] = 1  # B00
        W[c2][col] = 1  # C10
        col += 1
        
        # Term 3: P3
        U[0][col] = 1   # A00
        V[b1][col] = 1  # B01
        V[b3][col] = -1 # B11
        W[c1][col] = 1  # C01
        W[c3][col] = 1  # C11
        col += 1
        
        # Term 4: P4
        U[3][col] = 1   # A11
        V[b2][col] = 1  # B10
        V[b0][col] = -1 # B00
        W[c0][col] = 1  # C00
        W[c2][col] = 1  # C10
        col += 1
        
        # Term 5: P5
        U[0][col] = 1   # A00
        U[1][col] = 1   # A01
        V[b3][col] = 1  # B11
        W[c0][col] = -1 # C00
        W[c1][col] = 1  # C01
        col += 1
        
        # Term 6: P6
        U[2][col] = 1   # A10
        U[0][col] = -1  # A00
        V[b0][col] = 1  # B00
        V[b1][col] = 1  # B01
        W[c3][col] = 1  # C11
        col += 1
        
        # Term 7: P7
        U[3][col] = 1   # A11
        U[1][col] = -1  # A01
        V[b2][col] = 1  # B10
        V[b3][col] = 1  # B11
        W[c0][col] = 1  # C00
        col += 1
    
    # Process last block of width 1 (column 14)
    # B[0, 14] = index 14, B[1, 14] = index 29
    # C[0, 14] = index 14*2 + 0 = 28, C[1, 14] = index 14*2 + 1 = 29
    
    b0 = 14   # B[0, 14]
    b1 = 29   # B[1, 14]
    c0 = 28   # C[0, 14]
    c1 = 29   # C[1, 14]
    
    # Q1 = A00 * B00 -> C00
    U[0][col] = 1
    V[b0][col] = 1
    W[c0][col] = 1
    col += 1
    
    # Q2 = A01 * B10 -> C00
    U[1][col] = 1
    V[b1][col] = 1
    W[c0][col] = 1
    col += 1
    
    # Q3 = A10 * B00 -> C10
    U[2][col] = 1
    V[b0][col] = 1
    W[c1][col] = 1
    col += 1
    
    # Q4 = A11 * B10 -> C10
    U[3][col] = 1
    V[b1][col] = 1
    W[c1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
