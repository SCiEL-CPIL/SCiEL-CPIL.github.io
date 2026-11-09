"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Rank 49 for 4x4x4 (2-level Strassen) + rank 16 for 4x4x1 (schoolbook) = 65
    rank = 65
    
    # U: (n*m, rank) = (16, 65), row i*m+j = A[i,j]
    # V: (m*p, rank) = (20, 65), row j*p+k = B[j,k]  
    # W: (p*n, rank) = (20, 65), row k*n+i = C[k,i] (trace convention)
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    u2 = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    v2 = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    w2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    
    def add_subproblem(a_rows, b_rows, c_rows, col_offset):
        """Add a 2x2x2 Strassen sub-problem at the given column offset.
        a_rows: 4 indices in U for A00,A01,A10,A11
        b_rows: 4 indices in V for B00,B01,B10,B11
        c_rows: 4 indices in W for C00,C01,C10,C11
        """
        for col in range(7):
            for i in range(4):
                U[a_rows[i], col_offset + col] += u2[i, col]
                V[b_rows[i], col_offset + col] += v2[i, col]
                W[c_rows[i], col_offset + col] += w2[i, col]
    
    # 4x4x4 decomposition using block Strassen (rank 49)
    # A = [[A00, A01],[A10, A11]], each 2x2
    # B = [[B00, B01],[B10, B11]], each 2x2
    # C = A*B = [[C00, C01],[C10, C11]], each 2x2
    #
    # U row indexing: i*4+j for A[i,j]
    # V row indexing: j*4+k for B[j,k] (k=0..3 for the 4x4 part)
    # W row indexing: k*4+i for C[k,i]
    
    # Block row/col indices in the 4x4 matrices:
    # A00: rows 0-1, cols 0-1 -> U rows [0,1,4,5] (A[0,0],A[0,1],A[1,0],A[1,1])
    # A01: rows 0-1, cols 2-3 -> U rows [2,3,6,7]
    # A10: rows 2-3, cols 0-1 -> U rows [8,9,12,13]
    # A11: rows 2-3, cols 2-3 -> U rows [10,11,14,15]
    
    # B00: rows 0-1, cols 0-1 -> V rows [0,1,4,5] (B[0,0],B[0,1],B[1,0],B[1,1])
    # B01: rows 0-1, cols 2-3 -> V rows [2,3,6,7]
    # B10: rows 2-3, cols 0-1 -> V rows [8,9,12,13]
    # B11: rows 2-3, cols 2-3 -> V rows [10,11,14,15]
    
    # C00: rows 0-1, cols 0-1 -> W rows [0,1,4,5] (C[0,0],C[0,1],C[1,0],C[1,1])
    # C01: rows 0-1, cols 2-3 -> W rows [2,3,6,7]
    # C10: rows 2-3, cols 0-1 -> W rows [8,9,12,13]
    # C11: rows 2-3, cols 2-3 -> W rows [10,11,14,15]
    
    # Strassen formulas for 4x4:
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
    
    # For each Strassen multiplication, the "A" and "B" inputs are linear combinations
    # of 2x2 blocks. We handle this by adding contributions from each block.
    
    # Helper: define block row indices
    A00_rows = [0, 1, 4, 5]   # A[0,0], A[0,1], A[1,0], A[1,1]
    A01_rows = [2, 3, 6, 7]   # A[0,2], A[0,3], A[1,2], A[1,3]
    A10_rows = [8, 9, 12, 13] # A[2,0], A[2,1], A[3,0], A[3,1]
    A11_rows = [10, 11, 14, 15] # A[2,2], A[2,3], A[3,2], A[3,3]
    
    B00_rows = [0, 1, 4, 5]   # B[0,0], B[0,1], B[1,0], B[1,1]
    B01_rows = [2, 3, 6, 7]   # B[0,2], B[0,3], B[1,2], B[1,3]
    B10_rows = [8, 9, 12, 13] # B[2,0], B[2,1], B[3,0], B[3,1]
    B11_rows = [10, 11, 14, 15] # B[2,2], B[2,3], B[3,2], B[3,3]
    
    C00_rows = [0, 1, 4, 5]   # C[0,0], C[0,1], C[1,0], C[1,1]
    C01_rows = [2, 3, 6, 7]   # C[0,2], C[0,3], C[1,2], C[1,3]
    C10_rows = [8, 9, 12, 13] # C[2,0], C[2,1], C[3,0], C[3,1]
    C11_rows = [10, 11, 14, 15] # C[2,2], C[2,3], C[3,2], C[3,3]
    
    col = 0
    
    # M1 = (A00+A11)(B00+B11)
    # A input = A00 + A11, B input = B00 + B11
    # Add A00 contribution and A11 contribution to U
    # Add B00 contribution and B11 contribution to V
    # Output goes to C00, C01, C10, C11 with coefficients from w2
    # But we need to distribute: M1 contributes to C00, C01, C10, C11
    # Actually, M1 is a 2x2 matrix, and it contributes to all four C blocks.
    # Wait, no. M1 = (A00+A11)(B00+B11) is a single 2x2 matrix product.
    # Its result contributes to C00, C01, C10, C11? No.
    
    # Let me reconsider. In block Strassen:
    # C00 = M1 + M4 - M5 + M7  (each Mi is a 2x2 matrix)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So each Mi is a 2x2 matrix that gets added to specific C blocks.
    # For M1: it's a 2x2 matrix. It contributes +1 to C00 and +1 to C11.
    # Wait, that's not right either. Let me re-derive.
    
    # Standard Strassen for 2x2 block matrices:
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
    
    # So M1 contributes +1 to C00 and +1 to C11 (as 2x2 blocks)
    # M2 contributes +1 to C10 and -1 to C11
    # M3 contributes +1 to C01 and +1 to C11
    # M4 contributes +1 to C00 and +1 to C10
    # M5 contributes -1 to C00 and +1 to C01
    # M6 contributes +1 to C11
    # M7 contributes +1 to C00
    
    # For each Mi, it's a 2x2 matrix product. We decompose it using 2x2 Strassen (rank 7).
    # The 2x2 Strassen gives us a 2x2 output matrix.
    # We then add that 2x2 output to the appropriate C blocks.
    
    # So for M1 = (A00+A11)(B00+B11):
    # The "A" matrix is A00+A11 (2x2), "B" matrix is B00+B11 (2x2)
    # We decompose this 2x2x2 product using Strassen (7 terms)
    # The resulting 2x2 matrix M1 is added to C00 and C11
    
    # For the U part: A00+A11 means we add U contributions from A00 and A11
    # For the V part: B00+B11 means we add V contributions from B00 and B11
    # For the W part: M1 goes to C00 (coeff +1) and C11 (coeff +1)
    
    # Let me define a function that handles a Strassen multiplication where
    # A is a sum of blocks and B is a sum of blocks, and the result is distributed
    # to multiple C blocks with coefficients.
    
    def strassen_mult(a_block_list, b_block_list, c_block_coeff_list, col_offset):
        """
        a_block_list: list of (rows, coeff) for A blocks
        b_block_list: list of (rows, coeff) for B blocks  
        c_block_coeff_list: list of (rows, coeff) for C blocks
        """
        for col in range(7):
            # U contributions
            for rows, coeff in a_block_list:
                for i in range(4):
                    U[rows[i], col_offset + col] += u2[i, col] * coeff
            # V contributions
            for rows, coeff in b_block_list:
                for i in range(4):
                    V[rows[i], col_offset + col] += v2[i, col] * coeff
            # W contributions
            for rows, coeff in c_block_coeff_list:
                for i in range(4):
                    W[rows[i], col_offset + col] += w2[i, col] * coeff
    
    # M1 = (A00+A11)(B00+B11), result to C00(+1) and C11(+1)
    strassen_mult(
        [(A00_rows, 1), (A11_rows, 1)],
        [(B00_rows, 1), (B11_rows, 1)],
        [(C00_rows, 1), (C11_rows, 1)],
        col
    )
    col += 7
    
    # M2 = (A10+A11)B00, result to C10(+1) and C11(-1)
    strassen_mult(
        [(A10_rows, 1), (A11_rows, 1)],
        [(B00_rows, 1)],
        [(C10_rows, 1), (C11_rows, -1)],
        col
    )
    col += 7
    
    # M3 = A00(B01-B11), result to C01(+1) and C11(+1)
    strassen_mult(
        [(A00_rows, 1)],
        [(B01_rows, 1), (B11_rows, -1)],
        [(C01_rows, 1), (C11_rows, 1)],
        col
    )
    col += 7
    
    # M4 = A11(B10-B00), result to C00(+1) and C10(+1)
    strassen_mult(
        [(A11_rows, 1)],
        [(B10_rows, 1), (B00_rows, -1)],
        [(C00_rows, 1), (C10_rows, 1)],
        col
    )
    col += 7
    
    # M5 = (A00+A01)B11, result to C00(-1) and C01(+1)
    strassen_mult(
        [(A00_rows, 1), (A01_rows, 1)],
        [(B11_rows, 1)],
        [(C00_rows, -1), (C01_rows, 1)],
        col
    )
    col += 7
    
    # M6 = (A10-A00)(B00+B01), result to C11(+1)
    strassen_mult(
        [(A10_rows, 1), (A00_rows, -1)],
        [(B00_rows, 1), (B01_rows, 1)],
        [(C11_rows, 1)],
        col
    )
    col += 7
    
    # M7 = (A01-A11)(B10+B11), result to C00(+1)
    strassen_mult(
        [(A01_rows, 1), (A11_rows, -1)],
        [(B10_rows, 1), (B11_rows, 1)],
        [(C00_rows, 1)],
        col
    )
    col += 7
    
    # Now handle the 5th column (4x4x1 part) with schoolbook (rank 16)
    # C[:,4] = A @ B[:,4]
    # For each i (0-3) and j (0-3): C[i,4] += A[i,j] * B[j,4]
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1  # B[j,4]
            W[4 * n + i, col] = 1  # C[4,i] (k=4, i)
            col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
