"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    
    We use a Strassen-based block decomposition for the 4x4x4 part (rank 28) and extend it to 4x4x7.
    The approach:
    1. Decompose the 4x4x4 matrix multiplication using Strassen's algorithm (rank 28).
    2. For the 4x4x7 case, we treat B as two blocks: B_left (4x4) and B_right (4x3).
    3. Apply the Strassen decomposition to (A, B_left) -> C_left (4x4).
    4. For (A, B_right) -> C_right (4x3), we can use a similar approach but with fewer columns.
    
    Actually, let me think more carefully. The tensor is for multiplying (4x4) * (4x7) -> (4x7).
    The standard schoolbook rank is 4*4*7 = 112.
    
    Let's use a block approach:
    - Split B into B1 (4x4) and B2 (4x3).
    - Compute A*B1 using Strassen (rank 28) -> gives C1 (4x4)
    - Compute A*B2 using a modified approach (rank ?) -> gives C2 (4x3)
    
    For A*B2 where B2 is 4x3, we can think of this as computing 3 separate 4x4*4x1 products.
    Each 4x4*4x1 product has schoolbook rank 16, but we might do better.
    
    Alternatively, we can use the following approach:
    - The full problem is equivalent to computing 7 separate 4x4*4x1 matrix-vector products.
    - Each such product has schoolbook rank 16, so 7*16 = 112 total.
    
    But we can do better by sharing computations across the 7 vectors.
    
    Let me use a different strategy: use the Strassen decomposition for the 4x4x4 part (giving rank 28 for the left 4 columns of B), and then handle the remaining 3 columns of B separately.
    
    For the 3 additional columns, we can use a simple approach: each column of B (4x1) multiplied by A (4x4) gives a 4x1 result. The schoolbook rank for each is 16, so 3*16 = 48.
    
    Total rank: 28 + 48 = 76.
    
    But wait, we can potentially do better. Let me think about this more carefully.
    
    Actually, the cleanest approach is to use the Strassen decomposition for the entire 4x4x7 problem by viewing it as a block matrix multiplication.
    
    Let me construct the Strassen decomposition for 4x4x4 first, then extend it.
    
    For 4x4x4 Strassen:
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each sub-block is 2x2.
    Split B into 2x2 blocks: B = [[B00, B01], [B10, B11]] where each sub-block is 2x2.
    
    Strassen products:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    Each Mk is a 2x2 matrix product, so each has rank 8 (schoolbook for 2x2x2).
    Total rank: 7 * 8 = 56 for 4x4x4.
    
    Wait, that's not right. Each Mk is a 2x2 matrix product, which has schoolbook rank 4 (not 8). Let me reconsider.
    
    A 2x2 matrix product C = A*B where A and B are 2x2 has 4 entries in C, and the schoolbook rank is 4 (one multiplication per entry).
    
    So each Mk (which is a 2x2 product) has rank 4, and there are 7 such products, giving total rank 7*4 = 28 for 4x4x4. That's correct.
    
    Now for 4x4x7:
    Split B into B1 (4x4) and B2 (4x3).
    - A*B1: use Strassen, rank 28.
    - A*B2: B2 is 4x3. We can split B2 into columns: b0, b1, b2 (each 4x1).
      Each A*bj is a 4x4 * 4x1 product, schoolbook rank 16.
      Total for B2: 3 * 16 = 48.
    
    Total rank: 28 + 48 = 76.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We'll build U, V, W matrices with integer entries.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (28, rank)  
    # W has shape (p*n, rank) = (28, rank)
    
    # Strategy: 
    # Part 1: Strassen for A*B1 where B1 = B[:, 0:4] (4x4)
    # Part 2: Schoolbook for A*B2 where B2 = B[:, 4:7] (4x3)
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_column(u, v, w):
        nonlocal rank
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        rank += 1
    
    # Part 1: Strassen decomposition for 4x4 * 4x4 -> 4x4
    # Indices: A is 4x4, B1 is 4x4 (columns 0-3 of B), C1 is 4x4 (columns 0-3 of C)
    # 
    # A blocks (2x2 each):
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # B1 blocks (2x2 each):
    # B00 = B1[0:2, 0:2], B01 = B1[0:2, 2:4]
    # B10 = B1[2:4, 0:2], B11 = B1[2:4, 2:4]
    #
    # C1 blocks:
    # C00 = C1[0:2, 0:2] = M1 - M5 + M7 - M4 (need to verify)
    # Actually, standard Strassen:
    # C00 = M1 - M5 + M7 - M4... let me use the standard formulas.
    
    # Standard Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mk, which is a 2x2 product, we need 4 scalar multiplications.
    # Each scalar multiplication contributes to one entry of C.
    
    # Let's define the Strassen factors:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each Mk, the 2x2 product has 4 entries. Each entry (r,c) of Mk 
    # contributes to specific entries of C1.
    
    # Let me define a helper to add a 2x2 matrix product decomposition.
    # A 2x2 product X*Y where X and Y are 2x2 matrices.
    # X has entries x00, x01, x10, x11
    # Y has entries y00, y01, y10, y11
    # Product Z = X*Y has entries:
    # z00 = x00*y00 + x01*y10
    # z01 = x00*y01 + x01*y11
    # z10 = x10*y00 + x11*y10
    # z11 = x10*y01 + x11*y11
    
    # For each term like x_ij * y_kl, we create a rank-1 term.
    
    def add_2x2_product_decomp(x_coeffs, y_coeffs, c_block_row, c_block_col):
        """
        Add decomposition for a 2x2 product X*Y where X and Y are defined by linear combinations.
        x_coeffs: list of 4 lists, each with 4 (row, col, coeff) tuples for X's entries
        y_coeffs: similar for Y
        c_block_row: which 2x2 block row in C (0 or 1)
        c_block_col: which 2x2 block col in C (0 or 1)
        """
        # This is getting complex. Let me just enumerate all 4 scalar products.
        pass
    
    # Let me take a simpler approach and directly construct the rank-1 terms.
    
    # Define the Strassen factors as linear combinations of A and B entries.
    # A entry (i,j) maps to U index i*4+j
    # B entry (j,k) maps to V index j*7+k (since m=4, p=7)
    # C entry (i,k) maps to W index k*4+i (since n=4, p=7)
    
    # Strassen M1 = (A00 + A11) * (B00 + B11)
    # A00 + A11 has entries:
    # (0,0): a00 + a22, (0,1): a01 + a23
    # (1,0): a10 + a32, (1,1): a11 + a33
    # B00 + B11 has entries:
    # (0,0): b00 + b20, (0,1): b01 + b21
    # (1,0): b10 + b30, (1,1): b11 + b31
    
    # M1 is 2x2, product entries:
    # m1_00 = (a00+a22)*(b00+b20) + (a01+a23)*(b10+b30)
    # m1_01 = (a00+a22)*(b01+b21) + (a01+a23)*(b11+b31)
    # m1_10 = (a10+a32)*(b00+b20) + (a11+a33)*(b10+b30)
    # m1_11 = (a10+a32)*(b01+b21) + (a11+a33)*(b11+b31)
    
    # C1 reconstruction:
    # C00[0,0] = c00_00 = m1_00 - m5_00 + m7_00 - m4_00... 
    
    # This is getting very tedious to do by hand. Let me use a known rank-28 
    # decomposition for 4x4x4 and extend it.
    
    # Actually, let me just use the schoolbook for now but optimize where possible.
    # The minimum rank for 4x4x7 is unknown, but schoolbook is 112.
    # A simple improvement: use rank-8 for 2x2x2 blocks.
    
    # Let me use a block decomposition:
    # Split A into 2x2 blocks (2x2 each), B into 2x2 blocks (2x2 each) for first 4 cols.
    # For each 2x2x2 sub-problem, use rank 8 (schoolbook for 2x2x2 is 8).
    # There are 4 such sub-problems for the 4x4 part: 4*8 = 32.
    # For the remaining 3 columns of B: 3 separate 4x4*4x1, each rank 16: 3*16 = 48.
    # Total: 32 + 48 = 80.
    
    # That's better than 112. Let me implement this.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: 4x4 * 4x4 part (first 4 columns of B)
    # 4 sub-problems, each 2x2 * 2x2 -> 2x2
    # Sub-problem (bi, bj) where bi, bj in {0,1}:
    # A block: A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # B block: B[2*bj:2*bj+2, 2*bi:2*bi+2]  (since B is 4x4, rows 0-3, cols 0-3)
    # C block: C[2*bi:2*bi+2, 2*bj:2*bj+2]
    
    for bi in range(2):
        for bj in range(2):
            # A sub-block: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
            # B sub-block: rows 2*bj to 2*bj+1, cols 2*bi to 2*bi+1
            # C sub-block: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
            
            for ai in range(2):
                for ak in range(2):
                    for bk in range(2):
                        # A entry: row = 2*bi + ai, col = 2*bj + ak
                        a_row = 2 * bi + ai
                        a_col = 2 * bj + ak
                        # B entry: row = 2*bj + ak, col = 2*bi + bk
                        b_row = 2 * bj + ak
                        b_col = 2 * bi + bk
                        # C entry: row = 2*bi + ai, col = 2*bj + bk
                        c_row = 2 * bi + ai
                        c_col = 2 * bj + bk
                        
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        
                        u[a_row * m + a_col] = 1
                        v[b_row * p + b_col] = 1
                        w[c_col * n + c_row] = 1
                        
                        U_cols.append(u)
                        V_cols.append(v)
                        W_cols.append(w)
    
    # Part 2: 4x4 * 4x3 part (last 3 columns of B, cols 4-6)
    # 3 separate 4x4 * 4x1 products
    for col_offset in range(3):
        b_col = 4 + col_offset
        for i in range(n):
            for k in range(m):
                # A entry: (i, k)
                # B entry: (k, b_col)
                # C entry: (i, b_col)
                
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + k] = 1
                v[k * p + b_col] = 1
                w[b_col * n + i] = 1
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    
    # Convert lists of lists to proper format
    U = [list(col) for col in zip(*U_cols)] if U_cols else [[0] * rank for _ in range(n * m)]
    V = [list(col) for col in zip(*V_cols)] if V_cols else [[0] * rank for _ in range(m * p)]
    W = [list(col) for col in zip(*W_cols)] if W_cols else [[0] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
