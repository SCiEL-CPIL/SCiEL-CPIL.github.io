"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks to reduce rank from 50 to 46.
"""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_block(A, B):
    """
    Apply Strassen's algorithm to compute 2x2 matrix product using 7 multiplications.
    A and B are 2x2 matrices. Returns the 2x2 product.
    This is used conceptually; the actual decomposition below encodes the 7 terms.
    """
    # Standard Strassen formulas for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    pass  # Placeholder - we encode the decomposition directly below

def construct():
    """
    Returns a rank-46 exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Uses Strassen's algorithm on the 2x2 block structure (treating rows as 2x1 blocks
    and columns as 5 columns, grouping into 2x2 sub-blocks where possible).
    
    For 2x5 x 5x5: We can view this as multiplying a 2x5 by 5x5.
    Partition the 5 columns into 2+2+1 and the 5 rows of B into 2+2+1.
    Use Strassen on the 2x2 sub-blocks (4 such blocks), and schoolbook on the rest.
    
    Actually, let me think more carefully. The matrix is A (2x5) * B (5x5).
    We can partition A into blocks [A1 | A2 | A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1.
    Partition B into blocks [B1; B2; B3] where B1 is 2x5, B2 is 2x5, B3 is 1x5.
    
    Product = A1*B1 + A2*B2 + A3*B3
    
    For A1*B1: A1 is 2x2, B1 is 2x5. We can apply Strassen-like decomposition.
    Actually, let me use a different approach.
    
    Let me partition differently. View the 2x5 x 5x5 multiplication as:
    - Partition the inner dimension (5) into 2+2+1
    - A = [A_0 | A_1 | A_2] where A_0, A_1 are 2x2 and A_2 is 2x1
    - B = [B_0; B_1; B_2] where B_0, B_1 are 2x5 and B_2 is 1x5
    
    Product = A_0 @ B_0 + A_1 @ B_1 + A_2 @ B_2
    
    For A_0 @ B_0: 2x2 times 2x5. We can use Strassen on the 2x2 part.
    Write B_0 = [b0 | b1] where b0, b1 are 2x2... no, B_0 is 2x5.
    
    Let me try a cleaner approach. Partition both dimensions:
    A is 2x5, B is 5x5, result is 2x5.
    
    Partition A's columns: 2+2+1 -> A = [Aa | Ab | Ac], Aa, Ab are 2x2, Ac is 2x1
    Partition B's rows: 2+2+1 -> B = [Ba; Bb; Bc], Ba, Bb are 2x5, Bc is 1x5
    Partition B's columns: 2+2+1 -> Ba = [Baa | Bab | Bac], etc.
    
    This gets complex. Let me use a simpler known result.
    
    For 2xn x nxn, we can get rank 7n + 2(n-1) by applying Strassen to 2x2 blocks.
    For n=5: 7*5 + 2*4 = 35 + 8 = 43? That doesn't match.
    
    Let me just implement the direct Strassen-on-2x2-blocks approach:
    Treat the 2x5 x 5x5 as having the 2x2 sub-block structure in the first two columns/rows.
    
    Simplest correct approach: Use Strassen for the 2x2 x 2x2 sub-problems.
    Partition:
    - A (2x5) = [A12 | A3] where A12 is 2x2 (cols 0,1) and A3 is 2x3 (cols 2,3,4)
    - B (5x5) = [B12; B3] where B12 is 2x5 (rows 0,1) and B3 is 3x5 (rows 2,3,4)
    
    Product = A12 @ B12 + A3 @ B3
    
    A12 @ B12: 2x2 times 2x5. Write B12 = [B12a | B12b] where B12a is 2x2, B12b is 2x3.
    A12 @ B12a: 2x2 times 2x2 -> use Strassen (rank 7 instead of 4)
    A12 @ B12b: 2x2 times 2x3 -> schoolbook (rank 4*3=12)
    A3 @ B3: 2x3 times 3x5 -> schoolbook (rank 2*3*5=30)
    
    Total: 7 + 12 + 30 = 49. Not great.
    
    Better: Partition into more 2x2 blocks.
    A (2x5) = [A1 | A2 | A5] where A1, A2 are 2x2, A5 is 2x1
    B (5x5) = [B1; B2; B5] where B1, B2 are 2x5, B5 is 1x5
    Also partition B's columns: B1 = [B1a | B1b | B1c], etc.
    
    Let me just do the straightforward implementation that's known to work:
    For each 2x2 block of the 5x5 inner dimension, apply Strassen.
    
    Partition inner dimension 5 = 2 + 2 + 1
    A = [A_0 | A_1 | A_2], shapes: 2x2, 2x2, 2x1
    B = [B_0; B_1; B_2], shapes: 2x5, 2x5, 1x5
    
    For A_0 @ B_0 (2x2 x 2x5):
      Write B_0 = [B_0a | B_0b | B_0c] where B_0a is 2x2, B_0b is 2x2, B_0c is 2x1
      A_0 @ B_0a: Strassen 2x2x2 -> rank 7
      A_0 @ B_0b: Strassen 2x2x2 -> rank 7  
      A_0 @ B_0c: schoolbook 2x2x2x1 -> rank 4
      Subtotal: 7+7+4 = 18
    
    For A_1 @ B_1 (2x2 x 2x5):
      Similarly: 7+7+4 = 18
    
    For A_2 @ B_2 (2x1 x 1x5):
      Schoolbook: 2*1*5 = 10
    
    Total: 18 + 18 + 10 = 46
    
    This matches the expected rank of 46!
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # We'll build the decomposition column by column (each column = one rank-1 term)
    # Each term has: U_vec (length 10), V_vec (length 25), W_vec (length 10)
    # U index: i*m + j = i*5 + j for i in [0,1], j in [0,4]
    # V index: j*p + k = j*5 + k for j in [0,4], k in [0,4]  
    # W index: k*n + i = k*2 + i for k in [0,4], i in [0,1]
    
    terms = []  # Each term is (u_dict, v_dict, w_dict)
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. Entries are lists of (index, value) pairs."""
        terms.append((u_entries, v_entries, w_entries))
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        """
        Add schoolbook terms for a sub-block multiplication.
        a_shape = (nr, nc) for A block, b_shape = (nr2, nc2) for B block.
        a_rows, a_cols: row/col offsets in A
        b_rows, b_cols: row/col offsets in B
        out_rows, out_cols: row/col offsets in output
        """
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    # A[i, k] * B[k, j] contributes to output[i, j]
                    # A global index: (a_rows + i) * m + (a_cols + k)
                    # B global index: (b_rows + k) * p + (b_cols + j)
                    # W global index: (out_cols + j) * n + (out_rows + i)
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        """
        Add Strassen terms for 2x2 x 2x2 multiplication.
        A block at rows a_row_off..a_row_off+1, cols a_col_off..a_col_off+1
        B block at rows b_row_off..b_row_off+1, cols b_col_off..b_col_off+1
        Output at rows out_row_off..out_row_off+1, cols out_col_off..out_col_off+1
        """
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # Output[0,0] = M1 + M4 - M5 + M7
        # Output[0,1] = M3 + M5
        # Output[1,0] = M2 + M4
        # Output[1,1] = M1 - M2 + M3 + M6
        
        # A elements:
        # a = A[0,0] at (a_row_off, a_col_off)
        # b = A[0,1] at (a_row_off, a_col_off+1)
        # c = A[1,0] at (a_row_off+1, a_col_off)
        # d = A[1,1] at (a_row_off+1, a_col_off+1)
        
        # B elements:
        # e = B[0,0] at (b_row_off, b_col_off)
        # f = B[0,1] at (b_row_off, b_col_off+1)
        # g = B[1,0] at (b_row_off+1, b_col_off)
        # h = B[1,1] at (b_row_off+1, b_col_off+1)
        
        # Output elements:
        # [0,0] at (out_row_off, out_col_off)
        # [0,1] at (out_row_off, out_col_off+1)
        # [1,0] at (out_row_off+1, out_col_off)
        # [1,1] at (out_row_off+1, out_col_off+1)
        
        a_u = (a_row_off) * m + (a_col_off)
        b_u = (a_row_off) * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + (a_col_off)
        d_u = (a_row_off + 1) * m + (a_col_off + 1)
        
        e_v = (b_row_off) * p + (b_col_off)
        f_v = (b_row_off) * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + (b_col_off)
        h_v = (b_row_off + 1) * p + (b_col_off + 1)
        
        o00_w = (out_col_off) * n + (out_row_off)
        o01_w = (out_col_off + 1) * n + (out_row_off)
        o10_w = (out_col_off) * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)
        
        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])
    
    # Partition:
    # A (2x5) = [A_0 | A_1 | A_2]
    #   A_0: rows 0-1, cols 0-1 (2x2)
    #   A_1: rows 0-1, cols 2-3 (2x2)
    #   A_2: rows 0-1, col 4 (2x1)
    #
    # B (5x5) = [B_0; B_1; B_2]
    #   B_0: rows 0-1, cols 0-4 (2x5)
    #   B_1: rows 2-3, cols 0-4 (2x5)
    #   B_2: row 4, cols 0-4 (1x5)
    #
    # Further partition B_0 and B_1 columns:
    #   B_0 = [B_0a | B_0b | B_0c] where B_0a is 2x2 (cols 0-1), B_0b is 2x2 (cols 2-3), B_0c is 2x1 (col 4)
    #   B_1 = [B_1a | B_1b | B_1c] similarly
    
    # A_0 @ B_0:
    #   A_0 @ B_0a: Strassen 2x2x2 (A cols 0-1, B rows 0-1, B cols 0-1, output cols 0-1)
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    #   A_0 @ B_0b: Strassen 2x2x2 (A cols 0-1, B rows 0-1, B cols 2-3, output cols 2-3)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    #   A_0 @ B_0c: schoolbook 2x2 x 2x1 (A cols 0-1, B rows 0-1, B col 4, output col 4)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)
    
    # A_1 @ B_1:
    #   A_1 @ B_1a: Strassen 2x2x2 (A cols 2-3, B rows 2-3, B cols 0-1, output cols 0-1)
    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    #   A_1 @ B_1b: Strassen 2x2x2 (A cols 2-3, B rows 2-3, B cols 2-3, output cols 2-3)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    #   A_1 @ B_1c: schoolbook 2x2 x 2x1 (A cols 2-3, B rows 2-3, B col 4, output col 4)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)
    
    # A_2 @ B_2:
    #   schoolbook 2x1 x 1x5 (A col 4, B row 4, B cols 0-4, output cols 0-4)
    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)
    
    # Now convert terms to matrices
    total_rank = len(terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
    
    # Time guard
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
