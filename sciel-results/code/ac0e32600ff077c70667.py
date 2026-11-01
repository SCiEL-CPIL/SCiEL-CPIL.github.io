import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Improved decomposition: Rank 50
    # 6 Strassen blocks (2x2x2) and 2 Schoolbook blocks (2x2x1)
    R = 50
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    rank_offset = 0
    
    # Define Strassen terms for 2x2x2
    # u, v, w are local 4-element vectors (flattened 2x2)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]), # M3
        ([0, 0, 0, 1], [1, 0, -1, 0], [-1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, -1]) # M7
    ]
    
    # Define Schoolbook terms for 2x2x1
    # u, v are 4-element vectors, w is 2-element vector
    schoolbook_221_terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0]), # A00 B00 -> C0
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0]), # A01 B10 -> C0
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 1]), # A10 B00 -> C1
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 1])  # A11 B10 -> C1
    ]
    
    # Blocks: (A_row_start, A_col_start, B_row_start, B_col_start, B_width, type)
    # A is 2x4. B is 4x7.
    # A1: rows 0-1, cols 0-1. A2: rows 0-1, cols 2-3.
    # B1: rows 0-1. B2: rows 2-3.
    # B1 cols: 0-1 (w=2), 2-3 (w=2), 4-5 (w=2), 6 (w=1)
    # B2 cols: 0-1 (w=2), 2-3 (w=2), 4-5 (w=2), 6 (w=1)
    
    blocks = [
        # A1 * B1 blocks
        (0, 0, 0, 0, 2, 'strassen'), # A1 B11
        (0, 0, 0, 2, 2, 'strassen'), # A1 B12
        (0, 0, 0, 4, 2, 'strassen'), # A1 B13
        (0, 0, 0, 6, 1, 'schoolbook'), # A1 B14
        # A2 * B2 blocks
        (0, 2, 2, 0, 2, 'strassen'), # A2 B21
        (0, 2, 2, 2, 2, 'strassen'), # A2 B22
        (0, 2, 2, 4, 2, 'strassen'), # A2 B23
        (0, 2, 2, 6, 1, 'schoolbook')  # A2 B24
    ]
    
    for a_r, a_c, b_r, b_c, b_w, b_type in blocks:
        if b_type == 'strassen':
            terms = strassen_terms
            for u_local, v_local, w_local in terms:
                # Add to U
                # u_local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global U index: (a_r + i) * m + (a_c + j)
                for li in range(4):
                    if u_local[li] != 0:
                        i = li // 2
                        j = li % 2
                        gi = a_r + i
                        gj = a_c + j
                        U[gi * m + gj][rank_offset] += u_local[li]
                
                # Add to V
                # v_local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global V index: (b_r + i) * p + (b_c + j)
                for li in range(4):
                    if v_local[li] != 0:
                        i = li // 2
                        j = li % 2
                        gi = b_r + i
                        gj = b_c + j
                        V[gi * p + gj][rank_offset] += v_local[li]
                
                # Add to W
                # w_local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global W index: (b_c + j) * n + (a_r + i)
                for li in range(4):
                    if w_local[li] != 0:
                        i = li // 2
                        j = li % 2
                        gi = b_c + j
                        gj = a_r + i
                        W[gi * n + gj][rank_offset] += w_local[li]
                
                rank_offset += 1
        else: # schoolbook 2x2x1
            terms = schoolbook_221_terms
            for u_local, v_local, w_local in terms:
                # Add to U
                for li in range(4):
                    if u_local[li] != 0:
                        i = li // 2
                        j = li % 2
                        gi = a_r + i
                        gj = a_c + j
                        U[gi * m + gj][rank_offset] += u_local[li]
                
                # Add to V
                for li in range(4):
                    if v_local[li] != 0:
                        i = li // 2
                        j = li % 2
                        gi = b_r + i
                        gj = b_c + j
                        V[gi * p + gj][rank_offset] += v_local[li]
                
                # Add to W
                # w_local indices: 0:(0), 1:(1)
                # Global W index: (b_c + 0) * n + (a_r + i)
                for li in range(2):
                    if w_local[li] != 0:
                        i = li
                        gi = b_c # k is always 0 for 2x2x1
                        gj = a_r + i
                        W[gi * n + gj][rank_offset] += w_local[li]
                
                rank_offset += 1

    if time.time() > deadline:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
