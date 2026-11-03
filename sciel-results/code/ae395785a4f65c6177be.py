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
    # Dimensions
    n, m, p = 4, 4, 7
    
    # Total rank: 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3) = 97
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 (Rank 49) ---
    base_U, base_V, base_W = get_strassen_2x2_2x2_UVW()
    
    # Strassen steps definition for 4x4 blocks
    # Each step defines:
    # a_blocks: { (br, bc): coeff }
    # b_blocks: { (br, bc): coeff }
    # c_coeffs: { (br, bc): coeff } for adding result to C blocks
    
    steps = [
        # M1 = (A00+A11)(B00+B11)
        {
            'a': {(0,0): 1, (1,1): 1},
            'b': {(0,0): 1, (1,1): 1},
            'c': {(0,0): 1, (1,1): 1} # M1 adds to C00 and C11
        },
        # M2 = (A10+A11)B00
        {
            'a': {(1,0): 1, (1,1): 1},
            'b': {(0,0): 1},
            'c': {(1,0): 1} # M2 adds to C10
        },
        # M3 = A00(B01-B11)
        {
            'a': {(0,0): 1},
            'b': {(0,1): 1, (1,1): -1},
            'c': {(0,1): 1} # M3 adds to C01
        },
        # M4 = A11(B10-B00)
        {
            'a': {(1,1): 1},
            'b': {(1,0): 1, (0,0): -1},
            'c': {(0,0): 1, (1,0): 1} # M4 adds to C00 and C10
        },
        # M5 = (A00+A01)B11
        {
            'a': {(0,0): 1, (0,1): 1},
            'b': {(1,1): 1},
            'c': {(0,1): 1, (1,1): -1} # M5 adds to C01 and C11 (-1)
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'a': {(1,0): 1, (0,0): -1},
            'b': {(0,0): 1, (0,1): 1},
            'c': {(1,1): 1} # M6 adds to C11
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'a': {(0,1): 1, (1,1): -1},
            'b': {(1,0): 1, (1,1): 1},
            'c': {(0,0): 1, (1,1): -1} # M7 adds to C00 and C11 (-1)
        }
    ]
    
    # Helper to get global index for A/B elements
    def ab_idx(br, bc, lr, lc):
        # A is 4x4, B is 4x4 (left part)
        # Global row = br*2 + lr
        # Global col = bc*2 + lc
        # Index = row * m + col (for U, m=4)
        # For V, B is 4x7, but we only use left 4 cols.
        # B global row = br*2 + lr
        # B global col = bc*2 + lc (0..3)
        # Index = row * p + col (for V, p=7)
        return None # Placeholder, we calculate directly in loop
        
    # Iterate through Strassen steps
    for step_idx, step in enumerate(steps):
        for t in range(7): # Base Strassen term
            # Process U
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_u = base_U[local_idx][t]
                    if coeff_u == 0: continue
                    
                    # This local element is a sum of A blocks
                    for (br, bc), coeff_a in step['a'].items():
                        if coeff_a == 0: continue
                        # Global A index
                        row = br * 2 + lr
                        col = bc * 2 + lc
                        u_idx = row * m + col
                        U[u_idx][step_idx * 7 + t] += coeff_u * coeff_a
                        
            # Process V
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_v = base_V[local_idx][t]
                    if coeff_v == 0: continue
                    
                    # This local element is a sum of B blocks
                    for (br, bc), coeff_b in step['b'].items():
                        if coeff_b == 0: continue
                        # Global B index (left part, cols 0-3)
                        row = br * 2 + lr
                        col = bc * 2 + lc
                        v_idx = row * p + col
                        V[v_idx][step_idx * 7 + t] += coeff_v * coeff_b
                        
            # Process W
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_w = base_W[local_idx][t]
                    if coeff_w == 0: continue
                    
                    # This local element is a sum of C blocks
                    for (br, bc), coeff_c in step['c'].items():
                        if coeff_c == 0: continue
                        # Global C index
                        row = br * 2 + lr
                        col = bc * 2 + lc
                        # W index: k*n + i -> col*n + row
                        w_idx = col * n + row
                        W[w_idx][step_idx * 7 + t] += coeff_w * coeff_c

    # --- Part 2: Schoolbook 4x4x3 (Rank 48) ---
    # B_right is cols 4,5,6
    # C_right is cols 4,5,6
    rank_offset = 49
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # A[i, j]
                u_idx = i * m + j
                # B[j, 4+k]
                v_idx = j * p + (4 + k)
                # C[4+k, i]
                w_idx = (4 + k) * n + i
                
                U[u_idx][rank_offset + col] = 1
                V[v_idx][rank_offset + col] = 1
                W[w_idx][rank_offset + col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
