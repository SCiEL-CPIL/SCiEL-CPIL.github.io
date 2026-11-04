"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll use Strassen's algorithm for the 4x4x4 core, applied to each group of 4 columns.
    # For 4x4x4, we partition into 2x2 blocks and use Strassen (7 multiplications of 2x2 matrices).
    # Each 2x2x2 multiplication is done with Strassen (rank 7), giving total rank 7*7 = 49 per 4x4x4.
    # But we can do better: use schoolbook for inner 2x2x2 (rank 4), giving 7*4 = 28 per 4x4x4.
    # 
    # For 4x4x9:
    # - Columns 0-3: 4x4x4 with rank 28
    # - Columns 4-7: 4x4x4 with rank 28  
    # - Column 8: 4x4x1 with rank 16 (schoolbook)
    # Total rank: 28 + 28 + 16 = 72
    
    # Let's implement the 28-rank decomposition for 4x4x4.
    # 
    # The 4x4x4 Strassen decomposition:
    # Partition A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 matrix product. We compute each with schoolbook (4 scalar multiplications).
    # Total: 7 * 4 = 28 scalar multiplications.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    def add_4x4x4_strassen(col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        A is always the same 4x4 matrix (rows 0-15 in U).
        B block starts at b_offset in V (16 rows for 4x4).
        C block starts at c_offset in W (16 rows for 4x4).
        Returns: new column index
        """
        c = col_start
        
        # For 4x4 matrix A, row index in U = i*4 + j for A[i,j]
        # A11 = A[0:2, 0:2] -> U[0], U[1], U[4], U[5]
        # A12 = A[0:2, 2:4] -> U[2], U[3], U[6], U[7]
        # A21 = A[2:4, 0:2] -> U[8], U[9], U[12], U[13]
        # A22 = A[2:4, 2:4] -> U[10], U[11], U[14], U[15]
        
        # For B block at b_offset, row index in V = b_offset + i*4 + j for B[i,j]
        # B11 = B[0:2, 0:2] -> V[b_offset+0], V[b_offset+1], V[b_offset+4], V[b_offset+5]
        # B12 = B[0:2, 2:4] -> V[b_offset+2], V[b_offset+3], V[b_offset+6], V[b_offset+7]
        # B21 = B[2:4, 0:2] -> V[b_offset+8], V[b_offset+9], V[b_offset+12], V[b_offset+13]
        # B22 = B[2:4, 2:4] -> V[b_offset+10], V[b_offset+11], V[b_offset+14], V[b_offset+15]
        
        # For C block at c_offset, row index in W = c_offset + i*4 + j for C[i,j]
        # C11 = C[0:2, 0:2] -> W[c_offset+0], W[c_offset+1], W[c_offset+4], W[c_offset+5]
        # C12 = C[0:2, 2:4] -> W[c_offset+2], W[c_offset+3], W[c_offset+6], W[c_offset+7]
        # C21 = C[2:4, 0:2] -> W[c_offset+8], W[c_offset+9], W[c_offset+12], W[c_offset+13]
        # C22 = C[2:4, 2:4] -> W[c_offset+10], W[c_offset+11], W[c_offset+14], W[c_offset+15]
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22: U[0]+=1, U[1]+=1, U[4]+=1, U[5]+=1, U[10]+=1, U[11]+=1, U[14]+=1, U[15]+=1
        # B11+B22: V[b_offset+0]+=1, V[b_offset+1]+=1, V[b_offset+4]+=1, V[b_offset+5]+=1,
        #           V[b_offset+10]+=1, V[b_offset+11]+=1, V[b_offset+14]+=1, V[b_offset+15]+=1
        # M1 is a 2x2 matrix. Its entries contribute to C11 and C22.
        # M1[0,0] -> C11[0,0], C22[0,0]
        # M1[0,1] -> C11[0,1], C22[0,1]
        # M1[1,0] -> C11[1,0], C22[1,0]
        # M1[1,1] -> C11[1,1], C22[1,1]
        
        # But wait - each M_i is itself a 2x2 matrix product, which requires 4 scalar multiplications.
        # So M1 = (A11+A22) * (B11+B22) means we need 4 separate scalar products.
        
        # Let me reconsider. The 7 Strassen products are matrix products of 2x2 matrices.
        # To decompose into scalar multiplications, each 2x2x2 product needs its own decomposition.
        
        # For each M_i = X * Y where X and Y are 2x2 matrices:
        # We can use schoolbook: 4 scalar multiplications
        # Or Strassen: 7 scalar multiplications
        
        # Using schoolbook for each 2x2x2:
        # M_i[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
        # M_i[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
        # M_i[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
        # M_i[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
        
        # This gives 4 scalar multiplications per M_i, total 7*4 = 28.
        
        # However, the linear combinations in X and Y involve sums/differences of A and B entries.
        # We need to carefully track which U and V entries are used.
        
        # Let me define a helper that adds a 2x2x2 schoolbook decomposition.
        
        def add_2x2x2_schoolbook(a_rows, b_rows, c_rows, col):
            """
            Add schoolbook 2x2x2 decomposition.
            a_rows: list of 4 U row indices for the 2x2 A matrix [a00, a01, a10, a11]
            b_rows: list of 4 V row indices for the 2x2 B matrix [b00, b01, b10, b11]
            c_rows: list of 4 W row indices for the 2x2 C matrix [c00, c01, c10, c11]
            col: starting column
            Returns: new column
            """
            c = col
            # c00 = a00*b00 + a01*b10
            U[a_rows[0]][c] = 1
            V[b_rows[0]][c] = 1
            W[c_rows[0]][c] = 1
            c += 1
            U[a_rows[1]][c] = 1
            V[b_rows[2]][c] = 1
            W[c_rows[0]][c] = 1
            c += 1
            # c01 = a00*b01 + a01*b11
            U[a_rows[0]][c] = 1
            V[b_rows[1]][c] = 1
            W[c_rows[1]][c] = 1
            c += 1
            U[a_rows[1]][c] = 1
            V[b_rows[3]][c] = 1
            W[c_rows[1]][c] = 1
            c += 1
            # c10 = a10*b00 + a11*b10
            U[a_rows[2]][c] = 1
            V[b_rows[0]][c] = 1
            W[c_rows[2]][c] = 1
            c += 1
            U[a_rows[3]][c] = 1
            V[b_rows[2]][c] = 1
            W[c_rows[2]][c] = 1
            c += 1
            # c11 = a10*b01 + a11*b11
            U[a_rows[2]][c] = 1
            V[b_rows[1]][c] = 1
            W[c_rows[3]][c] = 1
            c += 1
            U[a_rows[3]][c] = 1
            V[b_rows[3]][c] = 1
            W[c_rows[3]][c] = 1
            c += 1
            return c
        
        # Now, for each of the 7 Strassen products, I need to define the linear combinations.
        # The problem is that the linear combinations create new "virtual" 2x2 matrices.
        # For example, M1 = (A11+A22)(B11+B22)
        # The first factor is a 2x2 matrix where:
        #   X[0,0] = A[0,0] + A[2,2]
        #   X[0,1] = A[0,1] + A[2,3]
        #   X[1,0] = A[1,0] + A[3,2]
        #   X[1,1] = A[1,1] + A[3,3]
        
        # This means in the U matrix, for this product, we need:
        # U row for X[0,0]: U[0] + U[10] (A[0,0] + A[2,2])
        # U row for X[0,1]: U[1] + U[11] (A[0,1] + A[2,3])
        # U row for X[1,0]: U[4] + U[14] (A[1,0] + A[3,2])
        # U row for X[1,1]: U[5] + U[15] (A[1,1] + A[3,3])
        
        # Similarly for B.
        
        # This is getting complex. Let me just hardcode the 28 scalar multiplications.
        
        # Actually, let me take a step back and use a simpler approach.
        # I'll use schoolbook for the entire 4x4x9, which gives rank 144.
        # The current code already does this. The score is 1.0.
        
        # To improve, I need to correctly implement the 28-rank decomposition.
        # Let me do it carefully.
        
        return c
    
    # Given the complexity, let me just use schoolbook for now.
    # The score will be 1.0.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
