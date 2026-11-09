"""Matrix multiplication 2,5,5 - Strassen block decomposition for rank 49."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-49 decomposition for 2x5 by 5x5 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2x2 block and schoolbook for the rest.
    Block structure:
      A is 2x5, B is 5x5, C is 2x5
      Split columns of A: A = [A1 | A2] where A1 is 2x2, A2 is 2x3
      Split rows of B: B = [B1; B2] where B1 is 2x5, B2 is 3x5
      Split columns of B: B1 = [B11 | B12] where B11 is 2x2, B12 is 2x3
                         B2 = [B21 | B22] where B21 is 3x2, B22 is 3x3
      C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    Ranks: 7 + 6 + 6 + 18 = 37? Wait, let me recalculate.
    
    Actually:
      A1*B11: 2x2 * 2x2 -> rank 7 (Strassen)
      A1*B12: 2x2 * 2x3 -> rank 12 (need to construct)
      A2*B21: 2x3 * 3x2 -> rank 12 (need to construct)
      A2*B22: 2x3 * 3x3 -> rank 18 (need to construct)
    Total: 7 + 12 + 12 + 18 = 49
    """
    # Let's use a simpler approach: just try to find any valid decomposition
    # with rank < 50 using known techniques.
    # 
    # For now, let's use the schoolbook (rank 50) as fallback, but try
    # a structured approach first.
    
    # Actually, let me think about this differently.
    # The simplest improvement: use Strassen for a 2x2x2 sub-problem.
    # 
    # Split the 5 columns into 2+3:
    # A (2x5) = [A_left (2x2) | A_right (2x3)]
    # B (5x5) rows split: top 2 rows, bottom 3 rows
    # B_top (2x5) = [B_tl (2x2) | B_tr (2x3)]
    # B_bot (3x5) = [B_bl (3x2) | B_br (3x3)]
    # 
    # C = A_left @ B_top + A_right @ B_bot
    #   = A_left @ B_tl + A_left @ B_tr + A_right @ B_bl + A_right @ B_br
    #
    # Ranks:
    #   A_left @ B_tl: 2x2 * 2x2 = 7 (Strassen)
    #   A_left @ B_tr: 2x2 * 2x3 = 12
    #   A_right @ B_bl: 2x3 * 3x2 = 12
    #   A_right @ B_br: 2x3 * 3x3 = 18
    # Total = 49
    #
    # I need explicit decompositions for each sub-problem.
    # Let me construct them.
    
    def strassen_222():
        """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
        # Strassen's algorithm: 7 multiplications
        # For A (2x2) * B (2x2):
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # Tensor form: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # A index: i*2+j (i=row of A, j=col of A)
        # B index: j*2+k (j=row of B, k=col of B)
        # C index: k*2+i (k=row of C, i=col of C)
        #
        # M1: (A00+A11)*(B00+B11)
        #   U: A00 (idx 0), A11 (idx 3)
        #   V: B00 (idx 0), B11 (idx 3)
        #   W: C00 (idx 0), C11 (idx 3)
        # M2: (A10+A11)*B00
        #   U: A10 (idx 2), A11 (idx 3)
        #   V: B00 (idx 0)
        #   W: C10 (idx 2), C11 (idx 3)
        # M3: A00*(B01-B11)
        #   U: A00 (idx 0)
        #   V: B01 (idx 1), B11 (idx 3) with -1
        #   W: C01 (idx 1), C11 (idx 3)
        # M4: A11*(B10-B00)
        #   U: A11 (idx 3)
        #   V: B10 (idx 2), B00 (idx 0) with -1
        #   W: C00 (idx 0), C11 (idx 3)
        # M5: (A00+A01)*B11
        #   U: A00 (idx 0), A01 (idx 1)
        #   V: B11 (idx 3)
        #   W: C00 (idx 0), C10 (idx 2)
        # M6: (A10-A00)*(B00+B01)
        #   U: A10 (idx 2), A00 (idx 0) with -1
        #   V: B00 (idx 0), B01 (idx 1)
        #   W: C11 (idx 3)
        # M7: (A01-A11)*(B10+B11)
        #   U: A01 (idx 1), A11 (idx 3) with -1
        #   V: B10 (idx 2), B11 (idx 3)
        #   W: C00 (idx 0), C01 (idx 1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3: A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: A11*(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[3][3] = 1
        
        # M5: (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6: (A10-A00)*(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)*(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        return U, V, W
    
    def schoolbook_223():
        """Return U, V, W for 2x2 * 2x3 with rank 12."""
        n, m, p = 2, 2, 3
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    def schoolbook_232():
        """Return U, V, W for 2x3 * 3x2 with rank 12."""
        n, m, p = 2, 3, 2
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    def schoolbook_233():
        """Return U, V, W for 2x3 * 3x3 with rank 18."""
        n, m, p = 2, 3, 3
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Now combine the four sub-decompositions
    # Overall: A is 2x5, B is 5x5, C is 2x5
    # A rows: 0,1; A cols: 0,1,2,3,4
    # B rows: 0,1,2,3,4; B cols: 0,1,2,3,4
    # C rows: 0,1; C cols: 0,1,2,3,4
    #
    # Split A cols: [0,1] and [2,3,4]
    # Split B rows: [0,1] and [2,3,4]
    # Split B cols: [0,1] and [2,3,4]
    #
    # Sub-problem 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (2x2 * 2x2, rank 7)
    # Sub-problem 2: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]  (2x2 * 2x3, rank 12)
    # Sub-problem 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]  (2x3 * 3x2, rank 12)
    # Sub-problem 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]  (2x3 * 3x3, rank 18)
    
    U1, V1, W1 = strassen_222()
    U2, V2, W2 = schoolbook_223()
    U3, V3, W3 = schoolbook_232()
    U4, V4, W4 = schoolbook_233()
    
    total_rank = 7 + 12 + 12 + 18
    U = [[0]*total_rank for _ in range(10)]  # 2*5 = 10
    V = [[0]*total_rank for _ in range(25)]  # 5*5 = 25
    W = [[0]*total_rank for _ in range(10)]  # 5*2 = 10
    
    # Sub-problem 1: offset 0, rank 7
    # A indices: i*2+j for i in 0..1, j in 0..1 -> A[0:2, 0:2] -> global A row i, col j
    #   Global A index: i*5 + j
    # B indices: j*2+k for j in 0..1, k in 0..1 -> B[0:2, 0:2] -> global B row j, col k
    #   Global B index: j*5 + k
    # C indices: k*2+i for k in 0..1, i in 0..1 -> C[0:2, 0:2] -> global C row k, col i
    #   Global C index: k*5 + i
    for c in range(7):
        for a_idx, val in enumerate(U1):
            if val != 0:
                # a_idx = i*2+j, i in 0..1, j in 0..1
                i, j = divmod(a_idx, 2)
                U[i*5 + j][c] += val
        for b_idx, val in enumerate(V1):
            if val != 0:
                # b_idx = j*2+k, j in 0..1, k in 0..1
                j, k = divmod(b_idx, 2)
                V[j*5 + k][c] += val
        for c_idx, val in enumerate(W1):
            if val != 0:
                # c_idx = k*2+i, k in 0..1, i in 0..1
                k, i = divmod(c_idx, 2)
                W[k*5 + i][c] += val
    
    # Sub-problem 2: offset 7, rank 12
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # A indices: i*2+j -> global A[i, j] -> U[i*5+j]
    # B indices: j*3+k where j in 0..1 (B row), k in 0..2 (B col offset by 2)
    #   Global B row j, col k+2 -> V[j*5 + (k+2)]
    # C indices: k*2+i where k in 0..2 (C col offset by 2), i in 0..1 (C row)
    #   Global C row i, col k+2 -> W[i*5 + (k+2)]
    offset = 7
    for c in range(12):
        for a_idx, val in enumerate(U2):
            if val != 0:
                i, j = divmod(a_idx, 2)
                U[i*5 + j][offset + c] += val
        for b_idx, val in enumerate(V2):
            if val != 0:
                # b_idx = j*3+k, j in 0..1, k in 0..2
                j, k = divmod(b_idx, 3)
                V[j*5 + (k+2)][offset + c] += val
        for c_idx, val in enumerate(W2):
            if val != 0:
                # c_idx = k*2+i, k in 0..2, i in 0..1
                k, i = divmod(c_idx, 2)
                W[i*5 + (k+2)][offset + c] += val
    
    # Sub-problem 3: offset 19, rank 12
    # A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # A indices: i*3+j where i in 0..1 (A row), j in 0..2 (A col offset by 2)
    #   Global A row i, col j+2 -> U[i*5 + (j+2)]
    # B indices: j*2+k where j in 0..2 (B row offset by 2), k in 0..1 (B col)
    #   Global B row j+2, col k -> V[(j+2)*5 + k]
    # C indices: k*2+i where k in 0..1 (C row), i in 0..1 (C col)
    #   Global C row k, col i -> W[k*5 + i]
    offset = 19
    for c in range(12):
        for a_idx, val in enumerate(U3):
            if val != 0:
                # a_idx = i*3+j, i in 0..1, j in 0..2
                i, j = divmod(a_idx, 3)
                U[i*5 + (j+2)][offset + c] += val
        for b_idx, val in enumerate(V3):
            if val != 0:
                # b_idx = j*2+k, j in 0..2, k in 0..1
                j, k = divmod(b_idx, 2)
                V[(j+2)*5 + k][offset + c] += val
        for c_idx, val in enumerate(W3):
            if val != 0:
                # c_idx = k*2+i, k in 0..1, i in 0..1
                k, i = divmod(c_idx, 2)
                W[k*5 + i][offset + c] += val
    
    # Sub-problem 4: offset 31, rank 18
    # A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # A indices: i*3+j where i in 0..1, j in 0..2
    #   Global A row i, col j+2 -> U[i*5 + (j+2)]
    # B indices: j*3+k where j in 0..2 (B row offset by 2), k in 0..2 (B col offset by 2)
    #   Global B row j+2, col k+2 -> V[(j+2)*5 + (k+2)]
    # C indices: k*2+i where k in 0..2 (C col offset by 2), i in 0..1 (C row)
    #   Global C row i, col k+2 -> W[i*5 + (k+2)]
    offset = 31
    for c in range(18):
        for a_idx, val in enumerate(U4):
            if val != 0:
                i, j = divmod(a_idx, 3)
                U[i*5 + (j+2)][offset + c] += val
        for b_idx, val in enumerate(V4):
            if val != 0:
                j, k = divmod(b_idx, 3)
                V[(j+2)*5 + (k+2)][offset + c] += val
        for c_idx, val in enumerate(W4):
            if val != 0:
                k, i = divmod(c_idx, 2)
                W[i*5 + (k+2)][offset + c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
