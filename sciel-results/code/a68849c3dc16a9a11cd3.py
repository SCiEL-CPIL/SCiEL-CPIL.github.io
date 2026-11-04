<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Rank-18 decomposition for 4x4x4 matrix multiplication
    # This is a known result: rank(4,4,4) = 18
    
    # I'll use a verified rank-18 decomposition.
    # The decomposition defines 18 scalar multiplications.
    
    # For 4x4x4, U is 16x18, V is 16x18, W is 16x18
    
    # Here's a known rank-18 decomposition (from literature):
    
    # Define the 18 products. Each product is of the form:
    # (linear combo of A entries) * (linear combo of B entries)
    # and contributes to C entries via linear combinations.
    
    # I'll use the following explicit rank-18 decomposition:
    
    # The 18 scalar multiplications:
    
    # Let me use a constructive approach based on known algorithms.
    
    # Actually, I found a valid rank-18 decomposition. Let me encode it.
    
    # U4: 16x18 matrix for A (4x4)
    # V4: 16x18 matrix for B (4x4)
    # W4: 16x18 matrix for C (4x4)
    
    # Using the decomposition that achieves rank 18:
    
    # I'll use the following (verified) rank-18 decomposition:
    
    # Product 1: (a00+a11+a22+a33) * (b00+b11+b22+b33)
    # This is getting too speculative. Let me use a safe approach.
    
    # Given the difficulty of hardcoding a correct rank-18 decomposition,
    # let me use a different strategy that is guaranteed correct:
    
    # Use the fact that we can do 4x4x4 with rank at most 18,
    # but for safety, use a decomposition I can verify.
    
    # I'll use the "Laderman" algorithm for 4x4x4 which has rank 20,
    # or use Strassen-based which has rank 28.
    
    # For Strassen-based 4x4x4:
    # Partition 4x4 into four 2x2 blocks
    # Use Strassen's 7 multiplications for the 2x2 block matrix multiplication
    # Each 2x2 multiplication is done with schoolbook (rank 4) or Strassen (rank 7)
    
    # If we use Strassen for the outer 2x2 block multiplication:
    # 7 scalar (2x2) multiplications
    # Each 2x2x2 can be done with rank 7 (Strassen)
    # Total rank: 7 * 7 = 49
    
    # That's better than 144!
    
    # Let me implement the Strassen-based decomposition for 4x4x4 with rank 49,
    # then use it for 8 columns (2 blocks) and schoolbook for 1 column.
    # Total: 49 + 49 + 16 = 114
    
    # Still better than 144.
    
    # But I can do better. Let me use schoolbook for the inner 2x2 multiplications.
    # 7 scalar (2x2) multiplications, each with rank 4 (schoolbook for 2x2x2)
    # Total rank: 7 * 4 = 28 for 4x4x4
    
    # For 9 columns: 28*2 + 16 = 72
    
    # Let me implement rank 28 for 4x4x4 using Strassen for outer, schoolbook for inner.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    # Helper: add Strassen 2x2x2 decomposition for a given block
    # For 2x2x2, Strassen uses 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_row_offset: row offset in U for the 2x2 A block (4 rows: a00,a01,a10,a11)
        b_row_offset: row offset in V for the 2x2 B block (4 rows: b00,b01,b10,b11)
        c_row_offset: row offset in W for the 2x2 C block (4 rows: c00,c01,c10,c11)
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        # A11 = a[0,0] -> U[a_row_offset + 0]
        # A22 = a[1,1] -> U[a_row_offset + 3]
        # B11 = b[0,0] -> V[b_row_offset + 0]
        # B22 = b[1,1] -> V[b_row_offset + 3]
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        # A21 = a[1,0] -> U[a_row_offset + 2]
        # A22 = a[1,1] -> U[a_row_offset + 3]
        # B11 = b[0,0] -> V[b_row_offset + 0]
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 2][c] = 1   # C21
        W[c_row_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        # A11 = a[0,0] -> U[a_row_offset + 0]
        # B21 = b[1,0] -> V[b_row_offset + 2]
        # B22 = b[1,1] -> V[b_row_offset + 3]
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = -1
        W[c_row_offset + 1][c] = 1   # C12
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        # A22 = a[1,1] -> U[a_row_offset + 3]
        # B12 = b[0,1] -> V[b_row_offset + 1]
        # B11 = b[0,0] -> V[b_row_offset + 0]
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 1][c] = 1
        V[b_row_offset + 0][c] = -1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        # A11 = a[0,0] -> U[a_row_offset + 0]
        # A12 = a[0,1] -> U[a_row_offset + 1]
        # B22 = b[1,1] -> V[b_row_offset + 3]
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 1][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = -1  # C11
        W[c_row_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21 = a[1,0] -> U[a_row_offset + 2]
        # A11 = a[0,0] -> U[a_row_offset + 0]
        # B11 = b[0,0] -> V[b_row_offset + 0]
        # B12 = b[0,1] -> V[b_row_offset + 1]
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 0][c] = -1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12 = a[0,1] -> U[a_row_offset + 1]
        # A22 = a[1,1] -> U[a_row_offset + 3]
        # B21 = b[1,0] -> V[b_row_offset + 2]
        # B22 = b[1,1] -> V[b_row_offset + 3]
        U[a_row_offset + 1][c] = 1
        U[a_row_offset + 3][c] = -1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    # Now for 4x4x4, partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen for the 2x2 block matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2 matrix product, which we compute with schoolbook (rank 4)
    
    # So for each M_i, we need 4 scalar multiplications.
    # Total: 7 * 4 = 28 scalar multiplications for 4x4x4.
    
    # Now I need to implement this for the 4x4x9 case.
    
    # For the first 4 columns of B (B[:, 0:4]):
    # Use the 28-rank decomposition
    # For columns 4:8:
    # Use another 28-rank decomposition
    # For column 8:
    # Use schoolbook: 4*4 = 16
    
    # Total rank: 28 + 28 + 16 = 72
    
    # Let me implement the 28-rank decomposition for 4x4x4
    
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        a_offset: row offset in U for the 4x4 A block (16 rows)
        b_offset: row offset in V for the 4x4 B block (16 rows)
        c_offset: row offset in W for the 4x4 C block (16 rows)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # The 7 Strassen products for 2x2 block matrix multiplication
        # Each is a 2x2 matrix product, computed with schoolbook (4 scalar mults)
        
        # M1 = (A11+A22)(B11+B22)
        # A11: rows a_offset+0:2, cols 0:2 -> U indices a_offset+0, a_offset+1
        # A22: rows a_offset+8:10, cols 8:10 -> U indices a_offset+8, a_offset+9
        # Wait, I need to be careful with indexing.
        
        # For 4x4 matrix A stored in U with row index = i*m + j = i*4 + j
        # A[i,j] is at U[a_offset + i*4 + j]
        
        # A11 = A[0:2, 0:2] -> U[a_offset+0], U[a_offset+1], U[a_offset+4], U[a_offset+5]
        # A12 = A[0:2, 2:4] -> U[a_offset+2], U[a_offset+3], U[a_offset+6], U[a_offset+7]
        # A21 = A[2:4, 0:2] -> U[a_offset+8], U[a_offset+9], U[a_offset+12], U[a_offset+13]
        # A22 = A[2:4, 2:4] -> U[a_offset+10], U[a_offset+11], U[a_offset+14], U[a_offset+15]
        
        # Similarly for B and C.
        
        # This is very complex. Let me simplify by using schoolbook for the whole thing
        # and just accept rank 144, since implementing the 28-rank correctly is error-prone.
        
        return c
    
    # Given the complexity and risk of errors, let me just use schoolbook.
    # The score will be 1.0, same as current.
    
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
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen's algorithm for the 4x4x4 core, then extend to 4x4x9.
    # 4x4x4 can be done with rank 28 (Strassen outer, schoolbook inner).
    # For 4x4x9: split B into two 4x4 blocks and one 4x1 block.
    # Total rank: 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    # Helper: add Strassen 2x2x2 decomposition for a given block
    def add_strassen_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_row_offset: row offset in U for the 2x2 A block (4 rows: a00,a01,a10,a11)
        b_row_offset: row offset in V for the 2x2 B block (4 rows: b00,b01,b10,b11)
        c_row_offset: row offset in W for the 2x2 C block (4 rows: c00,c01,c10,c11)
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 2][c] = 1   # C21
        W[c_row_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = -1
        W[c_row_offset + 1][c] = 1   # C12
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 1][c] = 1
        V[b_row_offset + 0][c] = -1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 1][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = -1  # C11
        W[c_row_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 0][c] = -1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_row_offset + 1][c] = 1
        U[a_row_offset + 3][c] = -1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    # Helper: add schoolbook 2x2x2 decomposition
    def add_schoolbook_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add schoolbook 2x2x2 decomposition (rank 4).
        """
        c = col_start
        # C00 = A00*B00
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 0][c] = 1
        c += 1
        # C01 = A00*B01
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 1][c] = 1
        c += 1
        # C10 = A10*B10
        U[a_row_offset + 2][c] = 1
        V[b_row_offset + 2][c] = 1
        W[c_row_offset + 2][c] = 1
        c += 1
        # C11 = A11*B11
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 3][c] = 1
        c += 1
        return c
    
    # Helper: add 4x4x4 decomposition using Strassen outer + schoolbook inner
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        a_offset: row offset in U for the 4x4 A block (16 rows)
        b_offset: row offset in V for the 4x4 B block (16 rows)
        c_offset: row offset in W for the 4x4 C block (16 rows)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # For 4x4 matrix A stored in U with row index = i*4 + j
        # A[i,j] is at U[a_offset + i*4 + j]
        
        # Partition A into 2x2 blocks:
        # A11 = A[0:2, 0:2] -> rows a_offset+0, a_offset+1, a_offset+4, a_offset+5
        # A12 = A[0:2, 2:4] -> rows a_offset+2, a_offset+3, a_offset+6, a_offset+7
        # A21 = A[2:4, 0:2] -> rows a_offset+8, a_offset+9, a_offset+12, a_offset+13
        # A22 = A[2:4, 2:4] -> rows a_offset+10, a_offset+11, a_offset+14, a_offset+15
        
        # Similarly for B and C.
        
        # Strassen formulas for 2x2 block matrix multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_i is a 2x2 matrix product, computed with schoolbook (4 scalar mults)
        
        # For each Strassen product, we need to:
        # 1. Form the linear combinations of A and B blocks
        # 2. Multiply the resulting 2x2 matrices with schoolbook
        
        # Let's define helper offsets for the 2x2 blocks within the 4x4:
        # For A:
        # A11: a_offset + [0, 1, 4, 5]
        # A12: a_offset + [2, 3, 6, 7]
        # A21: a_offset + [8, 9, 12, 13]
        # A22: a_offset + [10, 11, 14, 15]
        
        # For B:
        # B11: b_offset + [0, 1, 4, 5]
        # B12: b_offset + [2, 3, 6, 7]
        # B21: b_offset + [8, 9, 12, 13]
        # B22: b_offset + [10, 11, 14, 15]
        
        # For C:
        # C11: c_offset + [0, 1, 4, 5]
        # C12: c_offset + [2, 3, 6, 7]
        # C21: c_offset + [8, 9, 12, 13]
        # C22: c_offset + [10, 11, 14, 15]
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22: rows [0,1,4,5] + [10,11,14,15]
        # B11+B22: rows [0,1,4,5] + [10,11,14,15]
        # Result goes to C11 and C22
        
        # For schoolbook multiplication of two 2x2 matrices:
        # (X)(Y) where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
        # Z00 = x00*y00, Z01 = x00*y01, Z10 = x10*y10, Z11 = x11*y11
        
        # M1: X = A11+A22, Y = B11+B22
        # X00 = A[0,0] + A[2,2] -> U[a_offset+0], U[a_offset+10]
        # X01 = A[0,1] + A[2,3] -> U[a_offset+1], U[a_offset+11]
        # X10 = A[1,0] + A[3,2] -> U[a_offset+4], U[a_offset+14]
        # X11 = A[1,1] + A[3,3] -> U[a_offset+5], U[a_offset+15]
        # Y00 = B[0,0] + B[2,2] -> V[b_offset+0], V[b_offset+10]
        # Y01 = B[0,1] + B[2,3] -> V[b_offset+1], V[b_offset+11]
        # Y10 = B[1,0] + B[3,2] -> V[b_offset+4], V[b_offset+14]
        # Y11 = B[1,1] + B[3,3] -> V[b_offset+5], V[b_offset+15]
        
        # Z00 = X00*Y00 -> contributes to C[0,0] = C11[0,0] -> W[c_offset+0]
        U[a_offset+0][c] = 1
        U[a_offset+10][c] = 1
        V[b_offset+0][c] = 1
        V[b_offset+10][c] = 1
        W[c_offset+0][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> contributes to C[0,1] = C11[0,1] -> W[c_offset+1]
        U[a_offset+0][c] = 1
        U[a_offset+10][c] = 1
        V[b_offset+1][c] = 1
        V[b_offset+11][c] = 1
        W[c_offset+1][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> contributes to C[1,0] = C11[1,0] -> W[c_offset+4]
        U[a_offset+4][c] = 1
        U[a_offset+14][c] = 1
        V[b_offset+4][c] = 1
        V[b_offset+14][c] = 1
        W[c_offset+4][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> contributes to C[1,1] = C11[1,1] -> W[c_offset+5]
        # Also contributes to C22[1,1] -> W[c_offset+15]
        U[a_offset+5][c] = 1
        U[a_offset+15][c] = 1
        V[b_offset+5][c] = 1
        V[b_offset+15][c] = 1
        W[c_offset+5][c] = 1
        W[c_offset+15][c] = 1
        c += 1
        
        # M2 = (A21+A22)B11
        # A21+A22: 
        # A21[0,0]=A[2,0], A21[0,1]=A[2,1], A21[1,0]=A[3,0], A21[1,1]=A[3,1]
        # A22[0,0]=A[2,2], A22[0,1]=A[2,3], A22[1,0]=A[3,2], A22[1,1]=A[3,3]
        # X00 = A[2,0]+A[2,2] -> U[a_offset+8], U[a_offset+10]
        # X01 = A[2,1]+A[2,3] -> U[a_offset+9], U[a_offset+11]
        # X10 = A[3,0]+A[3,2] -> U[a_offset+12], U[a_offset+14]
        # X11 = A[3,1]+A[3,3] -> U[a_offset+13], U[a_offset+15]
        # B11:
        # Y00 = B[0,0] -> V[b_offset+0]
        # Y01 = B[0,1] -> V[b_offset+1]
        # Y10 = B[1,0] -> V[b_offset+4]
        # Y11 = B[1,1] -> V[b_offset+5]
        
        # Z00 = X00*Y00 -> C21[0,0] = C[2,0] -> W[c_offset+8]
        # Also C22[0,0] = C[2,2] -> W[c_offset+10] with -1
        U[a_offset+8][c] = 1
        U[a_offset+10][c] = 1
        V[b_offset+0][c] = 1
        W[c_offset+8][c] = 1
        W[c_offset+10][c] = -1
        c += 1
        
        # Z01 = X00*Y01 -> C21[0,1] = C[2,1] -> W[c_offset+9]
        # Also C22[0,1] = C[2,3] -> W[c_offset+11] with -1
        U[a_offset+8][c] = 1
        U[a_offset+10][c] = 1
        V[b_offset+1][c] = 1
        W[c_offset+9][c] = 1
        W[c_offset+11][c] = -1
        c += 1
        
        # Z10 = X10*Y10 -> C21[1,0] = C[3,0] -> W[c_offset+12]
        # Also C22[1,0] = C[3,2] -> W[c_offset+14] with -1
        U[a_offset+12][c] = 1
        U[a_offset+14][c] = 1
        V[b_offset+4][c] = 1
        W[c_offset+12][c] = 1
        W[c_offset+14][c] = -1
        c += 1
        
        # Z11 = X11*Y11 -> C21[1,1] = C[3,1] -> W[c_offset+13]
        # Also C22[1,1] = C[3,3] -> W[c_offset+15] with -1
        U[a_offset+13][c] = 1
        U[a_offset+15][c] = 1
        V[b_offset+5][c] = 1
        W[c_offset+13][c] = 1
        W[c_offset+15][c] = -1
        c += 1
        
        # M3 = A11(B21-B22)
        # A11:
        # X00 = A[0,0] -> U[a_offset+0]
        # X01 = A[0,1] -> U[a_offset+1]
        # X10 = A[1,0] -> U[a_offset+4]
        # X11 = A[1,1] -> U[a_offset+5]
        # B21-B22:
        # B21[0,0]=B[2,0], B21[0,1]=B[2,1], B21[1,0]=B[3,0], B21[1,1]=B[3,1]
        # B22[0,0]=B[2,2], B22[0,1]=B[2,3], B22[1,0]=B[3,2], B22[1,1]=B[3,3]
        # Y00 = B[2,0]-B[2,2] -> V[b_offset+8], V[b_offset+10]
        # Y01 = B[2,1]-B[2,3] -> V[b_offset+9], V[b_offset+11]
        # Y10 = B[3,0]-B[3,2] -> V[b_offset+12], V[b_offset+14]
        # Y11 = B[3,1]-B[3,3] -> V[b_offset+13], V[b_offset+15]
        
        # Z00 = X00*Y00 -> C12[0,0] = C[0,2] -> W[c_offset+2]
        # Also C22[0,0] = C[2,2] -> W[c_offset+10] with +1
        U[a_offset+0][c] = 1
        V[b_offset+8][c] = 1
        V[b_offset+10][c] = -1
        W[c_offset+2][c] = 1
        W[c_offset+10][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> C12[0,1] = C[0,3] -> W[c_offset+3]
        # Also C22[0,1] = C[2,3] -> W[c_offset+11] with +1
        U[a_offset+0][c] = 1
        V[b_offset+9][c] = 1
        V[b_offset+11][c] = -1
        W[c_offset+3][c] = 1
        W[c_offset+11][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> C12[1,0] = C[1,2] -> W[c_offset+6]
        # Also C22[1,0] = C[3,2] -> W[c_offset+14] with +1
        U[a_offset+4][c] = 1
        V[b_offset+12][c] = 1
        V[b_offset+14][c] = -1
        W[c_offset+6][c] = 1
        W[c_offset+14][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> C12[1,1] = C[1,3] -> W[c_offset+7]
        # Also C22[1,1] = C[3,3] -> W[c_offset+15] with +1
        U[a_offset+5][c] = 1
        V[b_offset+13][c] = 1
        V[b_offset+15][c] = -1
        W[c_offset+7][c] = 1
        W[c_offset+15][c] = 1
        c += 1
        
        # M4 = A22(B12-B11)
        # A22:
        # X00 = A[2,2] -> U[a_offset+10]
        # X01 = A[2,3] -> U[a_offset+11]
        # X10 = A[3,2] -> U[a_offset+14]
        # X11 = A[3,3] -> U[a_offset+15]
        # B12-B11:
        # B12[0,0]=B[0,2], B12[0,1]=B[0,3], B12[1,0]=B[1,2], B12[1,1]=B[1,3]
        # B11[0,0]=B[0,0], B11[0,1]=B[0,1], B11[1,0]=B[1,0], B11[1,1]=B[1,1]
        # Y00 = B[0,2]-B[0,0] -> V[b_offset+2], V[b_offset+0]
        # Y01 = B[0,3]-B[0,1] -> V[b_offset+3], V[b_offset+1]
        # Y10 = B[1,2]-B[1,0] -> V[b_offset+6], V[b_offset+4]
        # Y11 = B[1,3]-B[1,1] -> V[b_offset+7], V[b_offset+5]
        
        # Z00 = X00*Y00 -> C11[0,0] = C[0,0] -> W[c_offset+0] with +1
        # Also C21[0,0] = C[2,0] -> W[c_offset+8] with +1
        U[a_offset+10][c] = 1
        V[b_offset+2][c] = 1
        V[b_offset+0][c] = -1
        W[c_offset+0][c] = 1
        W[c_offset+8][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> C11[0,1] = C[0,1] -> W[c_offset+1] with +1
        # Also C21[0,1] = C[2,1] -> W[c_offset+9] with +1
        U[a_offset+10][c] = 1
        V[b_offset+3][c] = 1
        V[b_offset+1][c] = -1
        W[c_offset+1][c] = 1
        W[c_offset+9][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> C11[1,0] = C[1,0] -> W[c_offset+4] with +1
        # Also C21[1,0] = C[3,0] -> W[c_offset+12] with +1
        U[a_offset+14][c] = 1
        V[b_offset+6][c] = 1
        V[b_offset+4][c] = -1
        W[c_offset+4][c] = 1
        W[c_offset+12][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> C11[1,1] = C[1,1] -> W[c_offset+5] with +1
        # Also C21[1,1] = C[3,1] -> W[c_offset+13] with +1
        U[a_offset+15][c] = 1
        V[b_offset+7][c] = 1
        V[b_offset+5][c] = -1
        W[c_offset+5][c] = 1
        W[c_offset+13][c] = 1
        c += 1
        
        # M5 = (A11+A12)B22
        # A11+A12:
        # A11[0,0]=A[0,0], A11[0,1]=A[0,1], A11[1,0]=A[1,0], A11[1,1]=A[1,1]
        # A12[0,0]=A[0,2], A12[0,1]=A[0,3], A12[1,0]=A[1,2], A12[1,1]=A[1,3]
        # X00 = A[0,0]+A[0,2] -> U[a_offset+0], U[a_offset+2]
        # X01 = A[0,1]+A[0,3] -> U[a_offset+1], U[a_offset+3]
        # X10 = A[1,0]+A[1,2] -> U[a_offset+4], U[a_offset+6]
        # X11 = A[1,1]+A[1,3] -> U[a_offset+5], U[a_offset+7]
        # B22:
        # Y00 = B[2,2] -> V[b_offset+10]
        # Y01 = B[2,3] -> V[b_offset+11]
        # Y10 = B[3,2] -> V[b_offset+14]
        # Y11 = B[3,3] -> V[b_offset+15]
        
        # Z00 = X00*Y00 -> C11[0,0] = C[0,0] -> W[c_offset+0] with -1
        # Also C12[0,0] = C[0,2] -> W[c_offset+2] with +1
        U[a_offset+0][c] = 1
        U[a_offset+2][c] = 1
        V[b_offset+10][c] = 1
        W[c_offset+0][c] = -1
        W[c_offset+2][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> C11[0,1] = C[0,1] -> W[c_offset+1] with -1
        # Also C12[0,1] = C[0,3] -> W[c_offset+3] with +1
        U[a_offset+0][c] = 1
        U[a_offset+2][c] = 1
        V[b_offset+11][c] = 1
        W[c_offset+1][c] = -1
        W[c_offset+3][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> C11[1,0] = C[1,0] -> W[c_offset+4] with -1
        # Also C12[1,0] = C[1,2] -> W[c_offset+6] with +1
        U[a_offset+4][c] = 1
        U[a_offset+6][c] = 1
        V[b_offset+14][c] = 1
        W[c_offset+4][c] = -1
        W[c_offset+6][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> C11[1,1] = C[1,1] -> W[c_offset+5] with -1
        # Also C12[1,1] = C[1,3] -> W[c_offset+7] with +1
        U[a_offset+5][c] = 1
        U[a_offset+7][c] = 1
        V[b_offset+15][c] = 1
        W[c_offset+5][c] = -1
        W[c_offset+7][c] = 1
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21-A11:
        # A21[0,0]=A[2,0], A21[0,1]=A[2,1], A21[1,0]=A[3,0], A21[1,1]=A[3,1]
        # A11[0,0]=A[0,0], A11[0,1]=A[0,1], A11[1,0]=A[1,0], A11[1,1]=A[1,1]
        # X00 = A[2,0]-A[0,0] -> U[a_offset+8], U[a_offset+0]
        # X01 = A[2,1]-A[0,1] -> U[a_offset+9], U[a_offset+1]
        # X10 = A[3,0]-A[1,0] -> U[a_offset+12], U[a_offset+4]
        # X11 = A[3,1]-A[1,1] -> U[a_offset+13], U[a_offset+5]
        # B11+B12:
        # B11[0,0]=B[0,0], B11[0,1]=B[0,1], B11[1,0]=B[1,0], B11[1,1]=B[1,1]
        # B12[0,0]=B[0,2], B12[0,1]=B[0,3], B12[1,0]=B[1,2], B12[1,1]=B[1,3]
        # Y00 = B[0,0]+B[0,2] -> V[b_offset+0], V[b_offset+2]
        # Y01 = B[0,1]+B[0,3] -> V[b_offset+1], V[b_offset+3]
        # Y10 = B[1,0]+B[1,2] -> V[b_offset+4], V[b_offset+6]
        # Y11 = B[1,1]+B[1,3] -> V[b_offset+5], V[b_offset+7]
        
        # Z00 = X00*Y00 -> C22[0,0] = C[2,2] -> W[c_offset+10] with +1
        U[a_offset+8][c] = 1
        U[a_offset+0][c] = -1
        V[b_offset+0][c] = 1
        V[b_offset+2][c] = 1
        W[c_offset+10][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> C22[0,1] = C[2,3] -> W[c_offset+11] with +1
        U[a_offset+8][c] = 1
        U[a_offset+0][c] = -1
        V[b_offset+1][c] = 1
        V[b_offset+3][c] = 1
        W[c_offset+11][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> C22[1,0] = C[3,2] -> W[c_offset+14] with +1
        U[a_offset+12][c] = 1
        U[a_offset+4][c] = -1
        V[b_offset+4][c] = 1
        V[b_offset+6][c] = 1
        W[c_offset+14][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> C22[1,1] = C[3,3] -> W[c_offset+15] with +1
        U[a_offset+13][c] = 1
        U[a_offset+5][c] = -1
        V[b_offset+5][c] = 1
        V[b_offset+7][c] = 1
        W[c_offset+15][c] = 1
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12-A22:
        # A12[0,0]=A[0,2], A12[0,1]=A[0,3], A12[1,0]=A[1,2], A12[1,1]=A[1,3]
        # A22[0,0]=A[2,2], A22[0,1]=A[2,3], A22[1,0]=A[3,2], A22[1,1]=A[3,3]
        # X00 = A[0,2]-A[2,2] -> U[a_offset+2], U[a_offset+10]
        # X01 = A[0,3]-A[2,3] -> U[a_offset+3], U[a_offset+11]
        # X10 = A[1,2]-A[3,2] -> U[a_offset+6], U[a_offset+14]
        # X11 = A[1,3]-A[3,3] -> U[a_offset+7], U[a_offset+15]
        # B21+B22:
        # B21[0,0]=B[2,0], B21[0,1]=B[2,1], B21[1,0]=B[3,0], B21[1,1]=B[3,1]
        # B22[0,0]=B[2,2], B22[0,1]=B[2,3], B22[1,0]=B[3,2], B22[1,1]=B[3,3]
        # Y00 = B[2,0]+B[2,2] -> V[b_offset+8], V[b_offset+10]
        # Y01 = B[2,1]+B[2,3] -> V[b_offset+9], V[b_offset+11]
        # Y10 = B[3,0]+B[3,2] -> V[b_offset+12], V[b_offset+14]
        # Y11 = B[3,1]+B[3,3] -> V[b_offset+13], V[b_offset+15]
        
        # Z00 = X00*Y00 -> C11[0,0] = C[0,0] -> W[c_offset+0] with +1
        U[a_offset+2][c] = 1
        U[a_offset+10][c] = -1
        V[b_offset+8][c] = 1
        V[b_offset+10][c] = 1
        W[c_offset+0][c] = 1
        c += 1
        
        # Z01 = X00*Y01 -> C11[0,1] = C[0,1] -> W[c_offset+1] with +1
        U[a_offset+2][c] = 1
        U[a_offset+10][c] = -1
        V[b_offset+9][c] = 1
        V[b_offset+11][c] = 1
        W[c_offset+1][c] = 1
        c += 1
        
        # Z10 = X10*Y10 -> C11[1,0] = C[1,0] -> W[c_offset+4] with +1
        U[a_offset+6][c] = 1
        U[a_offset+14][c] = -1
        V[b_offset+12][c] = 1
        V[b_offset+14][c] = 1
        W[c_offset+4][c] = 1
        c += 1
        
        # Z11 = X11*Y11 -> C11[1,1] = C[1,1] -> W[c_offset+5] with +1
        U[a_offset+7][c] = 1
        U[a_offset+15][c] = -1
        V[b_offset+13][c] = 1
        V[b_offset+15][c] = 1
        W[c_offset+5][c] = 1
        c += 1
        
        return c
    
    # Helper: add schoolbook 4x4x1 decomposition (rank 16)
    def add_4x4x1_schoolbook(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add schoolbook 4x4x1 decomposition (rank 16).
        b_offset: row offset in V for the 4x1 B block (4 rows)
        c_offset: row offset in W for the 4x1 C block (4 rows)
        """
        c = col_start
        for i in range(4):
            for j in range(4):
                U[a_offset + i*4 + j][c] = 1
                V[b_offset + j][c] = 1
                W[c_offset + i][c] = 1
                c += 1
        return c
    
    # Now build the full 4x4x9 decomposition:
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # Split C into C1 (4x4), C2 (4x4), C3 (4x1)
    
    # For B1: b_offset = 0 (rows 0-3 of B, which are B[:, 0:4])
    # For B2: b_offset = 4 (rows 4-7 of B, which are B[:, 4:8])
    # For B3: b_offset = 8 (rows 8-11 of B, which are B[:, 8])
    
    # Wait, V has shape (m*p, rank) = (4*9, rank) = (36, rank)
    # B is 4x9, stored in V with row index = j*p + k = j*9 + k
    # B[j,k] is at V[j*9 + k]
    
    # So for B1 (columns 0-3): rows 0*9+0 to 3*9+3 = 0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30
    # For B2 (columns 4-7): rows 0*9+4 to 3*9+7 = 4,5,6,7, 13,14,15,16, 22,23,24,25, 31,32,33,34
    # For B3 (column 8): rows 0*9+8 to 3*9+8 = 8, 17, 26, 35
    
    # For C1 (columns 0-3): rows 0*4+0 to 3*4+3 = 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15
    # For C2 (columns 4-7): rows 0*4+4 to 3*4+7 = 4,5,6,7, 8,9,10,11, 12,13,14,15
    # Wait, W has shape (p*n, rank) = (9*4, rank) = (36, rank)
    # C is 9x4, stored in W with row index = k*n + i = k*4 + i
    # C[k,i] is at W[k*4 + i]
    
    # So for C1 (rows 0-3): rows 0*4+0 to 3*4+3 = 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15
    # For C2 (rows 4-7): rows 4*4+0 to 7*4+3 = 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31
    # For C3 (row 8): rows 8*4+0 to 8*4+3 = 32,33,34,35
    
    # A is 4x4, stored in U with row index = i*m + j = i*4 + j
    # A[i,j] is at U[i*4 + j]
    
    # Add 4x4x4 decomposition for B1 -> C1
    col = add_4x4x4_strassen(U, V, W, 0, 0, 0, col)
    
    # Add 4x4x4 decomposition for B2 -> C2
    col = add_4x4x4_strassen(U, V, W, 0, 0, 16, col)
    
    # Add 4x4x1 decomposition for B3 -> C3
    col = add_4x4x1_schoolbook(U, V, W, 0, 8, 32, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I need to fix the B offset issue. The V matrix stores B with row index = j*p + k = j*9 + k, so for the second 4x4 block (columns 4-7), the rows are not contiguous. Let me reconsider the approach.

Actually, looking at this more carefully, the Strassen-based approach for 4x4x4 is complex and error-prone. Let me use a simpler but still improved approach: use Strassen for 2x2x2 blocks and schoolbook for the rest.

Actually, let me reconsider. The issue is that B is 4x9, so when we split it into 4x4 blocks, the rows in V are not contiguous. Let me use a different approach: process each column of B independently using a 4x4x1 decomposition, but use Strassen for the 4x4 part when possible.

Actually, the cleanest approach is to use the fact that 4x4x4 can be done with rank 28 (Strassen outer + schoolbook inner), and then handle the 9 columns by splitting B into two 4x4 blocks and one 4x1 block. But we need to handle the non-contiguous rows in V.

Let me rewrite the solution more carefully:

