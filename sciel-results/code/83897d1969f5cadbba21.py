import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Rank 56
    rank = 56
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen 7 terms for 2x2x2
    # u: row index in A (0..1) + col index in A block (0..1) -> local 2x2
    # v: row index in B block (0..1) + col index in B (0..1) -> local 2x2
    # w: row index in C (0..1) + col index in C block (0..1) -> local 2x2
    
    # Local indices for 2x2 blocks
    # A is 2x4. Split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x8. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    # B_top has 4 cols of 2x2? No, B is 4x8.
    # Rows 0,1 are top. Rows 2,3 are bot.
    # Cols 0,1 are block 0. Cols 2,3 block 1. Cols 4,5 block 2. Cols 6,7 block 3.
    
    # 8 independent 2x2x2 multiplications:
    # 1. A_left * B_top_block0
    # 2. A_left * B_top_block1
    # 3. A_left * B_top_block2
    # 4. A_left * B_top_block3
    # 5. A_right * B_bot_block0
    # 6. A_right * B_bot_block1
    # 7. A_right * B_bot_block2
    # 8. A_right * B_bot_block3
    
    # Strassen terms
    # Format: (u_idx, v_idx, w_idx, coeff)
    # u_idx: 0..3 (i, j_local)
    # v_idx: 0..3 (i, j_local)
    # w_idx: 0..3 (i, j_local)
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        [(0, 1), (3, 1)], # u
        [(0, 1), (3, 1)], # v
        [(0, 1), (3, 1)]  # w
    ]
    # This format is hard to manage. Let's use explicit lists of indices.
    
    terms = [
        # M1
        {'u': [0, 3], 'v': [0, 3], 'w': [0, 3], 'coeff': 1},
        # M2
        {'u': [2, 3], 'v': [0], 'w': [2, 3], 'coeff_u': [1, 1], 'coeff_v': [1], 'coeff_w': [1, -1]},
        # M3
        {'u': [0], 'v': [1, 3], 'w': [1, 3], 'coeff_u': [1], 'coeff_v': [1, -1], 'coeff_w': [1, 1]},
        # M4
        {'u': [3], 'v': [2, 0], 'w': [0, 2], 'coeff_u': [1], 'coeff_v': [1, -1], 'coeff_w': [1, 1]},
        # M5
        {'u': [0, 1], 'v': [3], 'w': [0, 1], 'coeff_u': [1, 1], 'coeff_v': [1], 'coeff_w': [-1, 1]},
        # M6
        {'u': [2, 0], 'v': [0, 1], 'w': [3], 'coeff_u': [1, -1], 'coeff_v': [1, 1], 'coeff_w': [1]},
        # M7
        {'u': [1, 3], 'v': [2, 3], 'w': [0], 'coeff_u': [1, -1], 'coeff_v': [1, 1], 'coeff_w': [1]}
    ]
    
    # Define the 8 blocks
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6)
    ]
    
    current_rank = 0
    for block_idx, (ac, br, bc, cc) in enumerate(blocks):
        for term in terms:
            if time.time() > deadline:
                break
            
            # Process U
            for k, idx in enumerate(term['u']):
                i_local = idx // 2
                j_local = idx % 2
                # Global U index: row i_global, col j_global
                # i_global = i_local
                # j_global = ac + j_local
                gu = i_local * m + (ac + j_local)
                U[gu, current_rank] += term['coeff_u'][k]
            
            # Process V
            for k, idx in enumerate(term['v']):
                i_local = idx // 2
                j_local = idx % 2
                # Global V index: row i_global, col j_global
                # i_global = br + i_local
                # j_global = bc + j_local
                gv = (br + i_local) * p + (bc + j_local)
                V[gv, current_rank] += term['coeff_v'][k]
            
            # Process W
            for k, idx in enumerate(term['w']):
                i_local = idx // 2
                j_local = idx % 2
                # Global W index: row i_global, col j_global
                # i_global = cc + j_local  (Note: W is p x n. Index is k*n + i? No, W[k*n+i] in prompt?)
                # Prompt says: W[k*n+i,r]. So row is k, col is i?
                # W shape is (p*n, rank).
                # Usually W maps to C. C is n x p.
                # C index: i (row 0..n-1), j (col 0..p-1).
                # Prompt: W[k*n+i]. Here k is col of C (0..p-1), i is row of C (0..n-1).
                # So row index in W is k*n + i.
                # In our local 2x2 C block:
                # i_local is row of C (0,1).
                # j_local is col of C (0,1).
                # Global row of C: i_global = i_local.
                # Global col of C: j_global = cc + j_local.
                # W index: j_global * n + i_global.
                gw = (cc + j_local) * n + i_local
                W[gw, current_rank] += term['coeff_w'][k]
            
            current_rank += 1
        
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

