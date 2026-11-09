"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We'll construct a rank-53 decomposition:
    # - Rank-49 for [4,4,4] using 2-level Strassen
    # - Plus 4 terms for the 5th column
    
    # First, let's build the [4,4,4] rank-49 decomposition
    # Then extend to [4,4,5]
    
    # The approach: use block Strassen
    # A is 4x4, B is 4x4 (first 4 columns of the 4x5 B)
    # Split into 2x2 blocks
    
    # Let's define the decomposition symbolically
    # Each "scalar multiplication" in the Strassen decomposition corresponds to one rank-1 term
    
    # For the [4,4,4] part, we have 7 block multiplications, each of rank 7 (Strassen for 2x2)
    # Total: 7*7 = 49 terms
    
    # For the 5th column, we need 4 more terms (one for each row of A times the 5th column of B)
    # Total rank: 49 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 16 x 53
    V = [[0] * rank for _ in range(m * p)]  # 20 x 53
    W = [[0] * rank for _ in range(p * n)]  # 20 x 53
    
    # Helper: add a rank-1 term
    def add_term(i, j, u_val, j2, k2, v_val, k3, i3, w_val, col):
        """Add a term where U[i*m+j]=u_val, V[j2*p+k2]=v_val, W[k3*n+i3]=w_val"""
        U[i * m + j][col] = u_val
        V[j2 * p + k2][col] = v_val
        W[k3 * n + i3][col] = w_val
    
    # Build the [4,4,4] decomposition using 2-level Strassen
    # 
    # Level 1: 4x4 = 2x(2x2) blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    #
    # Strassen intermediate sums:
    # S1 = A00 + A11
    # S2 = A11 - A00
    # S3 = A00 + A01
    # S4 = A10 + A11
    # S5 = B00 + B11
    # S6 = B11 - B00
    # S7 = B00 + B01
    # S8 = B10 + B11
    #
    # M1 = S1 * S5
    # M2 = S4 * B00
    # M3 = S3 * S6
    # M4 = S2 * S7
    # M5 = A00 * S8
    # M6 = S2 * B01
    # M7 = S3 * B10
    #
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Each Mj is a 2x2 matrix product, decomposed with Strassen (7 scalar mults)
    
    # Let me define a function to decompose a 2x2 product using Strassen
    # and add the 7 terms to U, V, W
    
    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_row_offset, c_col_offset, col_start):
        """
        Decompose C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2] 
        = A[a_rows, a_cols] * B[b_rows, b_cols]
        where A is 2x2 with rows a_rows (list of 2 row indices in A) and cols a_cols
        B is 2x2 with rows b_rows and cols b_cols
        C is 2x2 at the given offset
        Returns col_start + 7
        """
        # A = [[A00, A01], [A10, A11]]
        # A00 is at A[a_rows[0], a_cols[0]]
        # A01 is at A[a_rows[0], a_cols[1]]
        # A10 is at A[a_rows[1], a_cols[0]]
        # A11 is at A[a_rows[1], a_cols[1]]
        
        # B = [[B00, B01], [B10, B11]]
        # B00 is at B[b_rows[0], b_cols[0]]
        # B01 is at B[b_rows[0], b_cols[1]]
        # B10 is at B[b_rows[1], b_cols[0]]
        # B11 is at B[b_rows[1], b_cols[1]]
        
        # C = [[C00, C01], [C10, C11]]
        # C00 is at C[c_row_offset, c_col_offset]
        # C01 is at C[c_row_offset, c_col_offset+1]
        # C10 is at C[c_row_offset+1, c_col_offset]
        # C11 is at C[c_row_offset+1, c_col_offset+1]
        
        col = col_start
        
        # Strassen terms for 2x2:
        # T1 = (A00 + A11) * (B00 + B11)
        # T2 = (A10 + A11) * B00
        # T3 = A00 * (B01 - B11)
        # T4 = A11 * (B10 - B00)
        # T5 = (A00 + A01) * B11
        # T6 = (A10 - A00) * (B00 + B01)
        # T7 = (A01 - A11) * (B10 + B11)
        
        # T1 = (A00 + A11) * (B00 + B11)
        # U: A00 and A11 contribute +1
        U[a_rows[0] * m + a_cols[0]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] += 1
        # V: B00 and B11 contribute +1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        # W: C00 += T1, C11 += T1
        W[(c_col_offset) * n + (c_row_offset)][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset+1)][col] += 1
        col += 1
        
        # T2 = (A10 + A11) * B00
        U[a_rows[1] * m + a_cols[0]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        W[(c_col_offset) * n + (c_row_offset+1)][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset+1)][col] += 1
        col += 1
        
        # T3 = A00 * (B01 - B11)
        U[a_rows[0] * m + a_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] -= 1
        W[(c_col_offset+1) * n + (c_row_offset)][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset+1)][col] -= 1
        col += 1
        
        # T4 = A11 * (B10 - B00)
        U[a_rows[1] * m + a_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[0]][col] -= 1
        W[(c_col_offset) * n + (c_row_offset+1)][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset+1)][col] -= 1
        col += 1
        
        # T5 = (A00 + A01) * B11
        U[a_rows[0] * m + a_cols[0]][col] += 1
        U[a_rows[0] * m + a_cols[1]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset)][col] += 1
        W[(c_col_offset+1) * n + (c_row_offset+1)][col] += 1
        col += 1
        
        # T6 = (A10 - A00) * (B00 + B01)
        U[a_rows[1] * m + a_cols[0]][col] += 1
        U[a_rows[0] * m + a_cols[0]][col] -= 1
        V[b_rows[0] * p + b_cols[0]][col] += 1
        V[b_rows[0] * p + b_cols[1]][col] += 1
        W[(c_col_offset) * n + (c_row_offset)][col] += 1
        W[(c_col_offset) * n + (c_row_offset+1)][col] += 1
        col += 1
        
        # T7 = (A01 - A11) * (B10 + B11)
        U[a_rows[0] * m + a_cols[1]][col] += 1
        U[a_rows[1] * m + a_cols[1]][col] -= 1
        V[b_rows[1] * p + b_cols[0]][col] += 1
        V[b_rows[1] * p + b_cols[1]][col] += 1
        W[(c_col_offset) * n + (c_row_offset)][col] -= 1
        W[(c_col_offset) * n + (c_row_offset+1)][col] -= 1
        col += 1
        
        return col
    
    # Now build the 4x4x4 decomposition
    # A is 4x4 with indices 0-3
    # B is 4x4 with indices 0-3 (first 4 columns of the 4x5 B)
    # C is 4x4 with indices 0-3
    
    # Block indices:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    col = 0
    
    # M1 = S1 * S5 where S1 = A00 + A11, S5 = B00 + B11
    # S1 is 2x2: [[A00_00+A11_00, A00_01+A11_01], [A00_10+A11_10, A00_11+A11_11]]
    # This is a 2x2 matrix with entries that are sums of A entries
    # S5 is 2x2: [[B00_00+B11_00, B00_01+B11_01], [B00_10+B11_10, B00_11+B11_11]]
    
    # For M1 = S1 * S5, I need to decompose this 2x2 product
    # But S1 and S5 are not simple 2x2 matrices with single entries - they have sums
    
    # This is where it gets complicated. The Strassen decomposition at the block level
    # requires that the intermediate sums are linear combinations, which means each
    # "scalar" in the 2x2 Strassen is actually a linear combination of multiple A or B entries.
    
    # Let me reconsider the approach. Instead of trying to do 2-level Strassen symbolically,
    # let me just directly construct the decomposition by tracking all the linear combinations.
    
    # Actually, I think the cleanest way is to just use numpy to construct the matrices
    # and verify correctness.
    
    # Let me try a different approach: use the known fact that rank([4,4,4]) <= 49
    # and construct it via the 2-level Strassen, but track all coefficients explicitly.
    
    # I'll build the decomposition by:
    # 1. For each of the 7 block products, use Strassen (7 scalar mults)
    # 2. Each "scalar" in the block-level Strassen is a linear combination of A or B entries
    # 3. Each "scalar" in the inner Strassen is a single A or B entry (or sum)
    
    # This is getting very complex. Let me just try to verify if my approach is correct
    # by constructing a small example and checking.
    
    # Given the complexity, let me fall back to a simpler approach:
    # Use the schoolbook decomposition but try to find a better one via search.
    
    # Actually, let me just return the schoolbook for now and note that this is a starting point.
    # The rank is 80.
    
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
