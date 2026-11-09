"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Strassen 2x2x2 factors:
    # U is 4x7: rows (A00, A01, A10, A11)
    # V is 4x7: rows (B00, B01, B10, B11)
    # W is 4x7: rows (C00, C01, C10, C11)
    #
    # Strassen products:
    # M1 = (A00+A10)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def strassen_factors():
        """Return U, V, W for a single 2x2x2 Strassen multiplication."""
        # U: 4 rows (A indices) x 7 cols (products)
        # Col 0: M1, Col 1: M2, Col 2: M3, Col 3: M4, Col 4: M5, Col 5: M6, Col 6: M7
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A10)*(B00+B10)
        U_s[0][0] = 1  # A00
        U_s[2][0] = 1  # A10
        V_s[0][0] = 1  # B00
        V_s[2][0] = 1  # B10
        W_s[0][0] = 1  # C00
        W_s[3][0] = 1  # C11
        
        # M2 = (A10+A11)*B00
        U_s[2][1] = 1  # A10
        U_s[3][1] = 1  # A11
        V_s[0][1] = 1  # B00
        W_s[2][1] = 1  # C10
        W_s[3][1] = -1  # C11
        
        # M3 = A00*(B01-B11)
        U_s[0][2] = 1  # A00
        V_s[1][2] = 1  # B01
        V_s[3][2] = -1  # B11
        W_s[1][2] = 1  # C01
        W_s[3][2] = 1  # C11
        
        # M4 = A11*(B10-B00)
        U_s[3][3] = 1  # A11
        V_s[2][3] = 1  # B10
        V_s[0][3] = -1  # B00
        W_s[0][3] = 1  # C00
        W_s[2][3] = 1  # C10
        
        # M5 = (A00+A01)*B11
        U_s[0][4] = 1  # A00
        U_s[1][4] = 1  # A01
        V_s[3][4] = 1  # B11
        W_s[0][4] = -1  # C00
        W_s[1][4] = 1  # C01
        
        # M6 = (A10-A00)*(B00+B01)
        U_s[2][5] = 1  # A10
        U_s[0][5] = -1  # A00
        V_s[0][5] = 1  # B00
        V_s[1][5] = 1  # B01
        W_s[3][5] = 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U_s[1][6] = 1  # A01
        U_s[3][6] = -1  # A11
        V_s[2][6] = 1  # B10
        V_s[3][6] = 1  # B11
        W_s[0][6] = 1  # C00
        
        return U_s, V_s, W_s
    
    U_s, V_s, W_s = strassen_factors()
    
    # For 2x4x6: 
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into A0 (cols 0-1) and A1 (cols 2-3)
    # Split B into B0 (rows 0-1, cols 0-5) and B1 (rows 2-3, cols 0-5)
    # C = A0*B0 + A1*B1
    # Each A*B product is 2x2 * 2x6 = 2x6
    # Split B0 and B1 further into three 2x2 blocks
    # So we have 6 sub-problems: A0*B0_j and A1*B1_j for j=0,1,2
    # Each sub-problem is 2x2 * 2x2, solved by Strassen (rank 7)
    # Total rank = 6 * 7 = 42
    
    rank = 42
    # U has n*m = 8 rows, rank columns
    # V has m*p = 24 rows, rank columns  
    # W has p*n = 12 rows, rank columns
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block indices:
    # A0 = A[:, 0:2], A1 = A[:, 2:4]
    # B0 = B[0:2, :], B1 = B[2:4, :]
    # B0 split into B0_0, B0_1, B0_2 (cols 0-1, 2-3, 4-5)
    # B1 split into B1_0, B1_1, B1_2 (cols 0-1, 2-3, 4-5)
    
    for block in range(6):
        a_idx = block // 3  # 0 or 1 (which A half)
        b_col_idx = block % 3  # 0, 1, or 2 (which B column block)
        b_row_idx = a_idx  # B0 uses rows 0-1, B1 uses rows 2-3
        
        col_offset = block * 7
        
        for j in range(4):
            for k in range(7):
                if U_s[j][k] != 0:
                    # A row index: a_idx*2 + j (A has 2 rows, each half has 2 cols)
                    # U row = i*m + j_local where i is A row, j_local is A col
                    # A col = a_idx*2 + j where j is 0 or 1 (local col in the 2-col half)
                    a_row = j // 2  # 0 or 1
                    a_col = a_idx*2 + j % 2  # 0-3
                    U[a_row*m + a_col][col_offset + k] = U_s[j][k]
                
                if V_s[j][k] != 0:
                    # B row index: b_row_idx*2 + j, B col = b_col_idx*2 + k_local
                    # V row = j*m + k where j is B row, k is B col
                    b_row = b_row_idx*2 + j//2
                    b_col = b_col_idx*2 + j%2
                    V[b_row*p + b_col][col_offset + k] = V_s[j][k]
                
                if W_s[j][k] != 0:
                    # C row = i (0 or 1), C col = b_col_idx*2 + j%2
                    # W index = k*n + i where k is C col, i is C row
                    c_row = j//2
                    c_col = b_col_idx*2 + j%2
                    W[c_col*n + c_row][col_offset + k] = W_s[j][k]
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        """Check that the decomposition exactly computes matrix multiplication."""
        # Build random test matrices
        import random
        random.seed(42)
        
        # Test with simple matrices
        for trial in range(3):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            
            # Compute AB directly
            AB = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        AB[i][k] += A[i][j] * B[j][k]
            
            # Compute using decomposition
            # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j
            # Actually: C = U^T * diag(W) * V^T doesn't work directly
            # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Where T[i,j,k] is the (i,k) element of A*B when A[i,j]*B[j,k]
            # So C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
            
            C_decomp = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for r in range(rank):
                            total += U[i*m + j][r] * A[i][j] * V[j*p + k][r] * B[j][k] * W[k*n + i][r]
                    C_decomp[i][k] = total
            
            if AB != C_decomp:
                return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][column] = 1
                    V[j*p + k][column] = 1
                    W[k*n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
