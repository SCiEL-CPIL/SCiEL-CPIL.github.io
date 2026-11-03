import time
import sys

# EVOLVE-BLOCK-START

def get_strassen_2x2_2x2_UVW():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen steps
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    #           0=B00, 1=B01, 2=B10, 3=B11
    #           0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 Strassen (Rank 49) for B columns 0-3
    # Part 2: 4x4x3 Schoolbook (Rank 48) for B columns 4-6
    rank_strassen = 49
    rank_schoolbook = 4 * 4 * 3  # 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x total_rank
    
    # --- Strassen 4x4x4 for B columns 0-3 ---
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Strassen 4x4x4: decompose A and B (4x4 part) into 2x2 blocks
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks (for cols 0-3): B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    #                          B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3)
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # The 7 Strassen formulas for 4x4x4:
    # Each step involves combining A blocks and B blocks, then applying 2x2 Strassen
    strassen_steps = [
        # (a_blocks, b_blocks) where blocks are {(br,bc): coeff}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # M7
    ]
    
    for step, (a_blocks, b_blocks) in enumerate(strassen_steps):
        for base_term in range(7):
            rank_idx = step * 7 + base_term
            
            # Fill U: for each local 2x2 position in the combined A matrix
            for lr in range(2):
                for lc in range(2):
                    local_a_idx = lr * 2 + lc
                    coeff_u = base_U[local_a_idx][base_term]
                    if coeff_u == 0:
                        continue
                    for (br, bc), coeff_a in a_blocks.items():
                        if coeff_a == 0:
                            continue
                        # Global A index: row = br*2+lr, col = bc*2+lc
                        global_a_row = br * 2 + lr
                        global_a_col = bc * 2 + lc
                        u_idx = global_a_row * m + global_a_col
                        U[u_idx][rank_idx] += coeff_u * coeff_a
            
            # Fill V: for each local 2x2 position in the combined B matrix
            for lr in range(2):
                for lc in range(2):
                    local_b_idx = lr * 2 + lc
                    coeff_v = base_V[local_b_idx][base_term]
                    if coeff_v == 0:
                        continue
                    for (br, bc), coeff_b in b_blocks.items():
                        if coeff_b == 0:
                            continue
                        # Global B index: row = br*2+lr, col = bc*2+lc
                        # B has m=4 rows, p=7 cols, but we're only using cols 0-3
                        global_b_row = br * 2 + lr
                        global_b_col = bc * 2 + lc
                        v_idx = global_b_row * p + global_b_col
                        V[v_idx][rank_idx] += coeff_v * coeff_b
            
            # Fill W: for each local 2x2 position in the C matrix
            for lr in range(2):
                for lc in range(2):
                    local_c_idx = lr * 2 + lc
                    coeff_w = base_W[local_c_idx][base_term]
                    if coeff_w == 0:
                        continue
                    # C block positions: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3)
                    #                    C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
                    # The output C has the same block structure as A and B
                    # For each (br, bc) in the result, add contribution
                    # C global: row = br*2+lr, col = bc*2+lc
                    # W index: k*n + i where k=col, i=row
                    for (br, bc) in [(0,0), (0,1), (1,0), (1,1)]:
                        # The C block at (br,bc) gets the local (lr,lc) element
                        global_c_row = br * 2 + lr
                        global_c_col = bc * 2 + lc
                        w_idx = global_c_col * n + global_c_row
                        W[w_idx][rank_idx] += coeff_w
    
    # --- Schoolbook 4x4x3 for B columns 4-6 ---
    offset = rank_strassen
    for i in range(n):  # row of A, row of C
        for j in range(m):  # col of A, row of B
            for k in range(3):  # col of B (4,5,6), col of C
                rank_idx = offset + (i * m + j) * 3 + k
                u_idx = i * m + j
                v_idx = j * p + (4 + k)  # B columns 4, 5, 6
                w_idx = (4 + k) * n + i  # C columns 4, 5, 6
                U[u_idx][rank_idx] = 1
                V[v_idx][rank_idx] = 1
                W[w_idx][rank_idx] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
