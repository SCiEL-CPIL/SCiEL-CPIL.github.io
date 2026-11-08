"""Matrix multiplication 2x4 by 4x6 tensor decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# Deadline guard: evaluator kills after 150s; we use 140s to be safe.
DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2 x 4
    # B is m x p = 4 x 6
    # C = A @ B is n x p = 2 x 6
    #
    # We partition A into two 2x2 blocks along the columns:
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    # We partition B into three 2x2 blocks along the rows (and corresponding columns of B):
    #   B1 = B[0:2, 0:2], B2 = B[2:4, 0:2]  -> for C[:, 0:2]
    #   B3 = B[0:2, 2:4], B4 = B[2:4, 2:4]  -> for C[:, 2:4]
    #   B5 = B[0:2, 4:6], B6 = B[2:4, 4:6]  -> for C[:, 4:6]
    #
    # C[:, 0:2] = A1 * B1 + A2 * B2
    # C[:, 2:4] = A1 * B3 + A2 * B4
    # C[:, 4:6] = A1 * B5 + A2 * B6
    #
    # Each 2x2 * 2x2 = 2x2 multiplication can be done with 7 terms (Strassen).
    # Total rank = 6 * 7 = 42.
    
    # Strassen 7-term algorithm for 2x2 matrices:
    # Given P (2x2) and Q (2x2), R = P * Q
    # P = [[p00, p01], [p10, p11]]
    # Q = [[q00, q01], [q10, q11]]
    #
    # M1 = (p00 + p11) * (q00 + q11)
    # M2 = (p10 + p11) * q00
    # M3 = p00 * (q01 - q11)
    # M4 = p11 * (q10 - q00)
    # M5 = (p00 + p01) * q11
    # M6 = (p10 - p00) * (q00 + q10)
    # M7 = (p01 - p11) * (q10 + q11)
    #
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form, each Strassen term corresponds to a rank-1 tensor.
    # For a single 2x2*2x2 multiplication, the 7 terms are:
    #
    # Term 1: (p00+p11) * (q00+q11) * (r00+r11)
    #   U: p00=1, p11=1; V: q00=1, q11=1; W: r00=1, r11=1
    # Term 2: (p10+p11) * q00 * (r10)
    #   U: p10=1, p11=1; V: q00=1; W: r10=1
    # Term 3: p00 * (q01-q11) * (r01+r11)
    #   U: p00=1; V: q01=1, q11=-1; W: r01=1, r11=1
    # Term 4: p11 * (q10-q00) * (r00+r10)
    #   U: p11=1; V: q10=1, q00=-1; W: r00=1, r10=1
    # Term 5: (p00+p01) * q11 * (r00+r01)
    #   U: p00=1, p01=1; V: q11=1; W: r00=1, r01=1
    # Term 6: (p10-p00) * (q00+q10) * (r11)
    #   U: p10=1, p00=-1; V: q00=1, q10=1; W: r11=1
    # Term 7: (p01-p11) * (q10+q11) * (r00+r11)
    #   U: p01=1, p11=-1; V: q10=1, q11=1; W: r00=1, r11=1
    
    # Define Strassen coefficients for a 2x2*2x2 multiplication.
    # Each term is a list of (index, coeff) for U, V, W.
    # U indices: 0=p00, 1=p01, 2=p10, 3=p11
    # V indices: 0=q00, 1=q01, 2=q10, 3=q11
    # W indices: 0=r00, 1=r01, 2=r10, 3=r11
    
    strassen_terms = [
        # (U: [(idx, coeff)], V: [(idx, coeff)], W: [(idx, coeff)])
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                   # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),          # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)]),  # M7
    ]
    
    rank = 6 * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Block assignments:
    # A blocks: A1 = A[:, 0:2] (cols 0,1), A2 = A[:, 2:4] (cols 2,3)
    # B blocks for each output column pair:
    #   C[:, 0:2]: B1 = B[0:2, 0:2], B2 = B[2:4, 0:2]
    #   C[:, 2:4]: B3 = B[0:2, 2:4], B4 = B[2:4, 2:4]
    #   C[:, 4:6]: B5 = B[0:2, 4:6], B6 = B[2:4, 4:6]
    #
    # For each pair (A_block, B_block, C_block), we apply Strassen.
    # A_block is 2x2, B_block is 2x2, C_block is 2x2.
    
    # Define the blocks:
    # (a_block_start_col, b_block_start_row, b_block_start_col, c_block_start_col)
    blocks = [
        (0, 0, 0, 0),  # A1 * B1 -> C[:, 0:2]
        (2, 2, 0, 0),  # A2 * B2 -> C[:, 0:2]
        (0, 0, 2, 2),  # A1 * B3 -> C[:, 2:4]
        (2, 2, 2, 2),  # A2 * B4 -> C[:, 2:4]
        (0, 0, 4, 4),  # A1 * B5 -> C[:, 4:6]
        (2, 2, 4, 4),  # A2 * B6 -> C[:, 4:6]
    ]
    
    for block_idx, (a_col_start, b_row_start, b_col_start, c_col_start) in enumerate(blocks):
        # For this block, A is A[:, a_col_start:a_col_start+2]
        # B is B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        # C is C[:, c_col_start:c_col_start+2]
        
        # U indices for this block's A (2x2):
        # A[i, j] where i in {0,1}, j in {0,1} relative to the block
        # Global A index: i * m + (a_col_start + j)
        # U local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # V indices for this block's B (2x2):
        # B[i, j] where i in {0,1}, j in {0,1} relative to the block
        # Global B index: (b_row_start + i) * p + (b_col_start + j)
        # V local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # W indices for this block's C (2x2):
        # C[i, j] where i in {0,1}, j in {0,1} relative to the block
        # Global C index: (c_col_start + j) * n + i  (W[k*n+i, r] where k is col, i is row)
        # W local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # Wait, W[k*n+i, r] where k is the column index of C, i is the row index.
        # So for C[i, j], the W index is j*n + i.
        # Local C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        # Global W index for local (i, j): (c_col_start + j) * n + i
        
        for term_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            col = block_idx * 7 + term_idx
            
            for (local_u_idx, coeff) in u_coeffs:
                i, j = divmod(local_u_idx, 2)  # i is row, j is col
                global_u_idx = i * m + (a_col_start + j)
                U[global_u_idx][col] += coeff
            
            for (local_v_idx, coeff) in v_coeffs:
                i, j = divmod(local_v_idx, 2)  # i is row, j is col
                global_v_idx = (b_row_start + i) * p + (b_col_start + j)
                V[global_v_idx][col] += coeff
            
            for (local_w_idx, coeff) in w_coeffs:
                i, j = divmod(local_w_idx, 2)  # i is row, j is col
                global_w_idx = (c_col_start + j) * n + i
                W[global_w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
