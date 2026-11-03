import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 decomposition (Rank 49)
    # Part 2: 4x4x1 decomposition (Rank 16)
    
    # Strassen 2x2x2 decomposition
    # u, v, w are vectors of length 4 (indices 0..3 for 2x2 matrix)
    # Index mapping: i*2+j
    s222 = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10 - C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3: A00(B01-B11) -> C01
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        # M4: A11(B10-B00) -> C10 + C00
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
        # M5: (A00+A01)B11 -> C01 - C11
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),
        # M6: (A10-A00)(B00+B01) -> C11
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00 + C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    # Outer Strassen for 4x4x4
    # Products P1..P7
    # Each P is a 2x2 matrix.
    # We define the A-blocks and B-blocks involved and their coefficients.
    # And the C-blocks they contribute to and their coefficients.
    
    # A blocks: 00, 01, 10, 11 (I, J)
    # B blocks: 00, 01, 10, 11 (J, K)
    # C blocks: 00, 01, 10, 11 (I, K)
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Structure: list of 7 dicts
    # Each dict has:
    # 'A': {block_idx: coeff}
    # 'B': {block_idx: coeff}
    # 'C': {block_idx: coeff}
    
    outer_strassen = [
        { # P1
            'A': {0: 1, 3: 1}, # A00, A11
            'B': {0: 1, 3: 1}, # B00, B11
            'C': {0: 1, 3: 1}  # C00, C11
        },
        { # P2
            'A': {2: 1, 3: 1}, # A10, A11
            'B': {0: 1},       # B00
            'C': {2: 1, 3: -1} # C10, C11
        },
        { # P3
            'A': {0: 1},       # A00
            'B': {1: 1, 3: -1},# B01, B11
            'C': {1: 1}        # C01
        },
        { # P4
            'A': {3: 1},       # A11
            'B': {2: 1, 0: -1},# B10, B00
            'C': {2: 1, 0: 1}  # C10, C00
        },
        { # P5
            'A': {0: 1, 1: 1}, # A00, A01
            'B': {3: 1},       # B11
            'C': {1: 1, 3: -1} # C01, C11
        },
        { # P6
            'A': {2: 1, 0: -1},# A10, A00
            'B': {0: 1, 1: 1}, # B00, B01
            'C': {3: 1}        # C11
        },
        { # P7
            'A': {1: 1, 3: -1},# A01, A11
            'B': {2: 1, 3: 1}, # B10, B11
            'C': {0: 1, 3: 1}  # C00, C11
        }
    ]
    
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    col = 0
    for p_idx in range(7):
        p_def = outer_strassen[p_idx]
        for s_idx in range(7):
            u_loc, v_loc, w_loc = s222[s_idx]
            
            # Expand U
            for l in range(4):
                u_val = u_loc[l]
                if u_val == 0: continue
                i_loc = l // 2
                j_loc = l % 2
                # For each A block involved in P
                for a_block, a_coeff in p_def['A'].items():
                    I = a_block // 2
                    J = a_block % 2
                    i_glob = I * 2 + i_loc
                    j_glob = J * 2 + j_loc
                    u_row = i_glob * 4 + j_glob
                    U_444[u_row][col] += u_val * a_coeff
            
            # Expand V
            for l in range(4):
                v_val = v_loc[l]
                if v_val == 0: continue
                j_loc = l // 2
                k_loc = l % 2
                for b_block, b_coeff in p_def['B'].items():
                    J = b_block // 2
                    K = b_block % 2
                    j_glob = J * 2 + j_loc
                    k_glob = K * 2 + k_loc
                    v_row = j_glob * 4 + k_glob
                    V_444[v_row][col] += v_val * b_coeff
            
            # Expand W
            for l in range(4):
                w_val = w_loc[l]
                if w_val == 0: continue
                k_loc = l // 2
                i_loc = l % 2
                for c_block, c_coeff in p_def['C'].items():
                    I = c_block // 2
                    K = c_block % 2
                    i_glob = I * 2 + i_loc
                    k_glob = K * 2 + k_loc
                    w_row = k_glob * 4 + i_glob
                    W_444[w_row][col] += w_val * c_coeff
            
            col += 1
    
    # Part 2: 4x4x1 decomposition (Rank 16)
    rank_441 = 16
    U_441 = [[0]*rank_441 for _ in range(16)]
    V_441 = [[0]*rank_441 for _ in range(4)] # m*p = 4*1 = 4
    W_441 = [[0]*rank_441 for _ in range(4)] # p*n = 1*4 = 4
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            u_row = i * 4 + j
            v_row = j * 1 + 0 # B is 4x1, index j*1+0
            w_row = 0 * 4 + i # C is 1x4, index 0*4+i
            
            U_441[u_row][col2] = 1
            V_441[v_row][col2] = 1
            W_441[w_row][col2] = 1
            col2 += 1
            
    # Combine
    total_rank = rank_444 + rank_441
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)] # 4*5
    W = [[0]*total_rank for _ in range(20)] # 5*4
    
    # Copy 444 part
    for r in range(16):
        for c in range(rank_444):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
            
    # Copy 441 part
    for r in range(16):
        for c in range(rank_441):
            U[r][rank_444 + c] = U_441[r][c]
            
    for r in range(4):
        for c in range(rank_441):
            # V rows 16-19 correspond to B2 (4x1)
            V[16 + r][rank_444 + c] = V_441[r][c]
            
    for r in range(4):
        for c in range(rank_441):
            # W rows 16-19 correspond to C2 (1x4)
            W[16 + r][rank_444 + c] = W_441[r][c]
            
    return {"U": U, "V": V, "W": W}
