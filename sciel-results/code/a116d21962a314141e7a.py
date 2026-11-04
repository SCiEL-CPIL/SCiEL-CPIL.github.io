import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Rank 56 decomposition for 2x4 * 4x8
    # This is achieved by using Strassen's algorithm (rank 7) on 8 independent 2x2x2 blocks.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    
    # The 7 products are:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # U (A part): 4x7
    # V (B part): 4x7
    # W (C part): 4x7
    
    u_coeffs = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=np.int32)
    
    v_coeffs = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 0, 0, 0, 1],  # b10
        [1, 0, 0, 1, 1, 0, 1]    # b11
    ], dtype=np.int32)
    
    w_coeffs = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 0, 0, 0, 1, 0]   # c11
    ], dtype=np.int32)
    
    # We have 8 blocks of 2x2x2.
    # The blocks are defined by:
    # - A block: rows 0-1, cols 0-1 or 2-3 (2 blocks)
    # - B block: rows 0-1 or 2-3, cols 0-1, 2-3, 4-5, 6-7 (4 blocks)
    # Total 2 * 4 = 8 blocks.
    
    for a_block in range(2):
        for b_block in range(4):
            base_r = (a_block * 4 + b_block) * 7
            
            # Map local indices (0..3) to global indices
            # A: 2x4 matrix. Flat index i*m + j.
            # A local (0,0) -> global (0, a_block*2)
            # A local (0,1) -> global (0, a_block*2 + 1)
            # A local (1,0) -> global (1, a_block*2)
            # A local (1,1) -> global (1, a_block*2 + 1)
            
            a_map = [
                0 * m + a_block * 2,
                0 * m + a_block * 2 + 1,
                1 * m + a_block * 2,
                1 * m + a_block * 2 + 1
            ]
            
            # B: 4x8 matrix. Flat index j*p + k.
            # B local (0,0) -> global (b_block*2, 0) ? No.
            # B is 4x8. Rows 0..3, Cols 0..7.
            # b_block 0 -> cols 0,1
            # b_block 1 -> cols 2,3
            # b_block 2 -> cols 4,5
            # b_block 3 -> cols 6,7
            # Rows are 0,1 for the top half of B (j=0,1) and 2,3 for the bottom half (j=2,3).
            # Wait, the split of m=4 is into j=0,1 and j=2,3.
            # So a_block 0 corresponds to j=0,1.
            # a_block 1 corresponds to j=2,3.
            
            # So for a_block 0:
            # B local (0,0) -> global (0, b_block*2)
            # B local (0,1) -> global (0, b_block*2 + 1)
            # B local (1,0) -> global (1, b_block*2)
            # B local (1,1) -> global (1, b_block*2 + 1)
            
            # For a_block 1:
            # B local (0,0) -> global (2, b_block*2)
            # B local (0,1) -> global (2, b_block*2 + 1)
            # B local (1,0) -> global (3, b_block*2)
            # B local (1,1) -> global (3, b_block*2 + 1)
            
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            
            b_map = [
                b_row_offset * p + b_col_offset,
                b_row_offset * p + b_col_offset + 1,
                (b_row_offset + 1) * p + b_col_offset,
                (b_row_offset + 1) * p + b_col_offset + 1
            ]
            
            # C: 2x8 matrix. Flat index k*n + i.
            # C local (0,0) -> global (b_block*2, 0) ?
            # C cols correspond to B cols.
            # C local (0,0) -> C[0, b_block*2] -> index (b_block*2)*2 + 0
            # C local (0,1) -> C[0, b_block*2+1] -> index (b_block*2+1)*2 + 0
            # C local (1,0) -> C[1, b_block*2] -> index (b_block*2)*2 + 1
            # C local (1,1) -> C[1, b_block*2+1] -> index (b_block*2+1)*2 + 1
            
            c_map = [
                (b_block * 2) * n + 0,
                (b_block * 2 + 1) * n + 0,
                (b_block * 2) * n + 1,
                (b_block * 2 + 1) * n + 1
            ]
            
            for l in range(7):
                r = base_r + l
                for local_idx in range(4):
                    coeff = u_coeffs[local_idx, l]
                    if coeff != 0:
                        U[a_map[local_idx], r] += coeff
                    coeff = v_coeffs[local_idx, l]
                    if coeff != 0:
                        V[b_map[local_idx], r] += coeff
                    coeff = w_coeffs[local_idx, l]
                    if coeff != 0:
                        W[c_map[local_idx], r] += coeff
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
