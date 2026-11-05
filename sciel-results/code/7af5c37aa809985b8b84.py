import time
import numpy as np

def construct():
    deadline = time.time() + 118.0
    
    n, m, p = 3, 4, 4
    
    # Rank-44 decomposition using 4 Strassen blocks + 16 schoolbook terms
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44
    
    # Strassen 2x2 terms
    strassen_terms = [
        # (u_dict, v_dict, w_dict) for 2x2x2
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_block(a_row_off, j_off, b_row_off, b_col_off, w_row_off, w_col_off, col_start):
        """Add a Strassen block for A[a_row_off:a_row_off+2, j_off:j_off+2] * B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
           contributing to C[w_row_off:w_row_off+2, w_col_off:w_col_off+2]"""
        for r in range(7):
            u_dict, v_dict, w_dict = strassen_terms[r]
            c = col_start + r
            for (ai, aj), val in u_dict.items():
                a_row = a_row_off + ai
                if a_row < n:
                    U[a_row * m + (j_off + aj)][c] += val
            for (bj, bk), val in v_dict.items():
                V[(b_row_off + bj) * p + (b_col_off + bk)][c] += val
            for (ci, ck), val in w_dict.items():
                w_row = w_row_off + ci
                if w_row < n:
                    W[(w_col_off + ck) * n + w_row][c] += val
        return col_start + 7
    
    col = 0
    # Four 2x2 Strassen blocks for the top 2 rows
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)      # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen_block(0, 2, 2, 0, 0, 2, col)      # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,2:4]  
    col = add_strassen_block(0, 0, 0, 2, 0, 0, col)      # Wait, this is wrong indexing
    
    # Let me reconsider the block structure.
    # A is 3x4, B is 4x4, C is 3x4
    # Split A into A_top (2x4) and A_bot (1x4)
    # Split B into B_left (4x2) and B_right (4x2)
    
    # C_top_left (2x2) = A_top (2x4) * B_left (4x2)
    #   = A_top[0:2,0:2] * B_left[0:2,0:2] + A_top[0:2,2:4] * B_left[2:4,0:2]
    #   Use Strassen for each 2x2 * 2x2: 7 + 7 = 14
    
    # C_top_right (2x2) = A_top (2x4) * B_right (4x2)
    #   = A_top[0:2,0:2] * B_right[0:2,0:2] + A_top[0:2,2:4] * B_right[2:4,0:2]
    #   Use Strassen: 7 + 7 = 14
    
    # C_bot_left (1x2) = A_bot (1x4) * B_left (4x2)
    #   = A_bot[0,0:2] * B_left[0:2,0:2] + A_bot[0,2:4] * B_left[2:4,0:2]
    #   Schoolbook: 2 + 2 = 4
    
    # C_bot_right (1x2) = A_bot (1x4) * B_right (4x2)
    #   = A_bot[0,0:2] * B_right[0:2,0:2] + A_bot[0,2:4] * B_right[2:4,0:2]
    #   Schoolbook: 2 + 2 = 4
    
    # Total: 14 + 14 + 4 + 4 = 36? That seems too good.
    
    # Wait, I need to be more careful. The Strassen decomposition computes a 2x2 * 2x2 product.
    # For C_top_left = A_top * B_left where A_top is 2x4 and B_left is 4x2:
    # This is NOT a 2x2 * 2x2 product. It's a 2x4 * 4x2 product.
    # To use Strassen, I need to split further.
    
    # A_top (2x4) = [A_tl (2x2) | A_tr (2x2)]
    # B_left (4x2) = [B_tl (2x2); B_bl (2x2)]
    # C_top_left (2x2) = A_tl * B_tl + A_tr * B_bl
    # Each 2x2*2x2 uses Strassen rank 7, so total 14. Correct.
    
    # Similarly for others.
    
    # Let me redo the col counter.
    col = 0
    
    # C_top_left (2x2): A_top[0:2,0:2] * B_left[0:2,0:2] + A_top[0:2,2:4] * B_left[2:4,0:2]
    # Term 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen_block(a_row_off=0, j_off=0, b_row_off=0, b_col_off=0, w_row_off=0, w_col_off=0, col_start=col)
    # Term 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    col = add_strassen_block(a_row_off=0, j_off=2, b_row_off=2, b_col_off=0, w_row_off=0, w_col_off=0, col_start=col)
    
    # C_top_right (2x2): A_top[0:2,0:2] * B_right[0:2,0:2] + A_top[0:2,2:4] * B_right[2:4,0:2]
    # Term 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen_block(a_row_off=0, j_off=0, b_row_off=0, b_col_off=2, w_row_off=0, w_col_off=2, col_start=col)
    # Term 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    col = add_strassen_block(a_row_off=0, j_off=2, b_row_off=2, b_col_off=2, w_row_off=0, w_col_off=2, col_start=col)
    
    # C_bot_left (1x2): A_bot[0,0:2] * B_left[0:2,0:2] + A_bot[0,2:4] * B_left[2:4,0:2]
    # Schoolbook: 4 terms
    for j in range(2):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    for j in range(2, 4):
        for k in range(2):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # C_bot_right (1x2): A_bot[0,0:2] * B_right[0:2,0:2] + A_bot[0,2:4] * B_right[2:4,0:2]
    # Schoolbook: 4 terms
    for j in range(2):
        for k in range(2, 4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    for j in range(2, 4):
        for k in range(2, 4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # col should be 28 + 8 = 36? But total_rank is 44. Let me check.
    # 4 Strassen blocks: 4*7 = 28
    # 8 schoolbook: 8
    # Total: 36
    
    # That's better than 44! But is it correct?
    # The issue: in the Strassen blocks, I'm using the same Strassen decomposition,
    # but the w_dict contributions might overlap incorrectly.
    
    # Let me verify by checking if col == 36 and adjusting total_rank.
    if col != total_rank:
        # Resize
        new_U = [row[:col] for row in U]
        new_V = [row[:col] for row in V]
        new_W = [row[:col] for row in W]
        U, V, W = new_U, new_V, new_W
        total_rank = col
    
    # Verify
    def _verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None
    
    valid, err = _verify(U, V, W, total_rank)
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook rank 48
    rank48 = n * m * p
    U48 = [[0] * rank48 for _ in range(n * m)]
    V48 = [[0] * rank48 for _ in range(m * p)]
    W48 = [[0] * rank48 for _ in range(p * n)]
    col48 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U48[i * m + j][col48] = 1
                V48[j * p + k][col48] = 1
                W48[k * n + i][col48] = 1
                col48 += 1
    return {"U": U48, "V": V48, "W": W48}
