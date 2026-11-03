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
    
    # Part 1: 4x4x4 Strassen (Rank 49) for first 4 columns of B
    # Part 2: 4x4x3 Schoolbook (Rank 48) for remaining 3 columns of B
    
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 4x4x4 for first 4 columns ---
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Strassen 4x4x4: decompose into 2x2 blocks, apply 7 Strassen steps
    # Each step uses the 2x2x2 Strassen decomposition
    # A blocks: A00(rows 0-1, cols 0-1), A01(rows 0-1, cols 2-3), A10(rows 2-3, cols 0-1), A11(rows 2-3, cols 2-3)
    # B blocks: B00(rows 0-1, cols 0-1), B01(rows 0-1, cols 2-3), B10(rows 2-3, cols 0-1), B11(rows 2-3, cols 2-3)
    # C blocks: C00(rows 0-1, cols 0-1), C01(rows 0-1, cols 2-3), C10(rows 2-3, cols 0-1), C11(rows 2-3, cols 2-3)
    
    # Define the 7 Strassen steps with their A and B block contributions
    strassen_steps = [
        # step, a_blocks: {(br,bc): coeff}, b_blocks: {(br,bc): coeff}
        (0, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),       # M1
        (1, {(1,0): 1, (1,1): 1}, {(0,0): 1}),                    # M2
        (2, {(0,0): 1}, {(0,1): 1, (1,1): -1}),                   # M3
        (3, {(1,1): 1}, {(1,0): 1, (0,0): -1}),                   # M4
        (4, {(0,0): 1, (0,1): 1}, {(1,1): 1}),                    # M5
        (5, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),         # M6
        (6, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),         # M7
    ]
    
    for step_idx, a_blocks, b_blocks in strassen_steps:
        for base_term in range(7):
            rank_col = step_idx * 7 + base_term
            
            # Fill U: for each local 2x2 A element, sum over contributing A blocks
            for lr in range(2):
                for lc in range(2):
                    local_a_idx = lr * 2 + lc
                    coeff_u = base_U[local_a_idx][base_term]
                    if coeff_u == 0:
                        continue
                    for (br, bc), coeff_a in a_blocks.items():
                        if coeff_a == 0:
                            continue
                        global_row = br * 2 + lr
                        global_col = bc * 2 + lc
                        u_idx = global_row * m + global_col
                        U[u_idx][rank_col] += coeff_u * coeff_a
            
            # Fill V: for each local 2x2 B element, sum over contributing B blocks
            for lr in range(2):
                for lc in range(2):
                    local_b_idx = lr * 2 + lc
                    coeff_v = base_V[local_b_idx][base_term]
                    if coeff_v == 0:
                        continue
                    for (br, bc), coeff_b in b_blocks.items():
                        if coeff_b == 0:
                            continue
                        global_row = br * 2 + lr
                        global_col = bc * 2 + lc
                        # B is 4x4 for this part (first 4 columns)
                        v_idx = global_row * 4 + global_col
                        V[v_idx][rank_col] += coeff_v * coeff_b
            
            # Fill W: for each local 2x2 C element, map to global C index
            # W index: k*n + i where k is col (0-3 for this part), i is row (0-3)
            for lr in range(2):
                for lc in range(2):
                    local_c_idx = lr * 2 + lc
                    coeff_w = base_W[local_c_idx][base_term]
                    if coeff_w == 0:
                        continue
                    global_row = lr
                    global_col = lc
                    # For C in 4x4x4, rows 0-3, cols 0-3
                    # W index = k*n + i = col * 4 + row
                    w_idx = global_col * n + global_row
                    W[w_idx][rank_col] += coeff_w
    
    # --- Schoolbook for remaining 3 columns (cols 4,5,6 of B) ---
    t = 49
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
