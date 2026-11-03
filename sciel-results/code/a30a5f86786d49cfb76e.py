"""Matrix multiplication 2,4,6 — Rank-42 Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-42 decomposition of the 2x4 * 4x6 matrix multiplication tensor.
    
    Strategy:
    1. Split the multiplication into two parts based on the columns of B:
       C_left = A * B_left (2x4 * 4x4)
       C_right = A * B_right (2x4 * 4x2)
    2. For C_left (2x4 * 4x4):
       Split A into two 2x2 blocks A1, A2.
       Split B_left into two 2x4 blocks B1, B2 (top/bottom rows).
       C_left = A1*B1 + A2*B2.
       Each term is 2x2 * 2x4.
       2x2 * 2x4 is two 2x2 * 2x2 multiplications.
       Each 2x2 * 2x2 uses Strassen (rank 7).
       So 2x2 * 2x4 has rank 14.
       Total rank for C_left = 14 + 14 = 28.
    3. For C_right (2x4 * 4x2):
       Split A into A1, A2 (2x2).
       Split B_right into B1, B2 (2x2).
       C_right = A1*B1 + A2*B2.
       Each term is 2x2 * 2x2 (rank 7).
       Total rank for C_right = 7 + 7 = 14.
    4. Total rank = 28 + 14 = 42.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 * 2x2
    # A local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Output indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_U = [
        {0: 1, 3: 1},       # p1 = (A00+A11)
        {2: 1, 3: 1},       # p2 = (A10+A11)
        {0: 1},             # p3 = A00
        {3: 1},             # p4 = A11
        {0: 1, 1: 1},       # p5 = (A00+A01)
        {2: 1, 0: -1},      # p6 = (A10-A00)
        {1: 1, 3: -1},      # p7 = (A01-A11)
    ]
    strassen_V = [
        {0: 1, 3: 1},       # p1 * (B00+B11)
        {0: 1},             # p2 * B00
        {1: 1, 3: -1},      # p3 * (B01-B11)
        {2: 1, 0: -1},      # p4 * (B10-B00)
        {3: 1},             # p5 * B11
        {0: 1, 1: 1},       # p6 * (B00+B01)
        {2: 1, 3: 1},       # p7 * (B10+B11)
    ]
    
    result_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = p1 + p4 - p5 + p7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = p3 + p5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = p2 + p4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = p1 - p2 + p3 + p6
    ]

    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_row_offset, w_col_offset, current_col):
        for s in range(7):
            # Add to U
            for a_idx, coeff in strassen_U[s].items():
                a_i = a_idx // 2
                a_j_local = a_idx % 2
                a_j_global = a_col_offset + a_j_local
                U[a_row_offset + a_i, a_j_global][current_col] += coeff
                
            # Add to V
            for b_idx, coeff in strassen_V[s].items():
                b_j_local = b_idx // 2
                b_k_local = b_idx % 2
                b_j_global = b_row_offset + b_j_local
                b_k_global = b_col_offset + b_k_local
                V[b_j_global, b_k_global][current_col] += coeff
                
            # Add to W (convention: W[k*n + i, r])
            for out_idx in range(4):
                coeff = result_coeffs[out_idx][s]
                if coeff != 0:
                    out_i = out_idx // 2
                    out_k_local = out_idx % 2
                    out_k_global = w_col_offset + out_k_local
                    W[out_k_global * n + out_i][current_col] += coeff
            current_col += 1
        return current_col

    # Part 1: C_left (2x4 * 4x4) -> Rank 28
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-3)
    # B1 is 2x4, so it's two 2x2 blocks: B1_left (cols 0-1), B1_right (cols 2-3)
    
    # A1 * B1_left (2x2 * 2x2)
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    # A1 * B1_right (2x2 * 2x2)
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    
    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-3)
    # B2 is 2x4, so it's two 2x2 blocks: B2_left (cols 0-1), B2_right (cols 2-3)
    
    # A2 * B2_left (2x2 * 2x2)
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    # A2 * B2_right (2x2 * 2x2)
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    
    # Part 2: C_right (2x4 * 4x2) -> Rank 14
    # A1 (rows 0-1, cols 0-1) * B1_right_part (rows 0-1, cols 4-5)
    
    # A1 * B1_right_part (2x2 * 2x2)
    col = add_strassen_block(0, 0, 0, 4, 0, 4, col)
    
    # A2 (rows 0-1, cols 2-3) * B2_right_part (rows 2-3, cols 4-5)
    
    # A2 * B2_right_part (2x2 * 2x2)
    col = add_strassen_block(0, 2, 2, 4, 0, 4, col)
    
    if time.time() > deadline:
        # Return best effort (which is the complete static construction)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
