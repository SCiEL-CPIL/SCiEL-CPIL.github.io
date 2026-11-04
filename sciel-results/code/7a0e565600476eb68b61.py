import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 77
    rank = 77
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Indices for A (4x4), B (4x6), C (4x6)
    # A[i, j] -> index i*m + j
    # B[j, k] -> index j*p + k
    # C[i, k] -> index k*n + i (Trace convention W[k*n+i, r])
    
    def idx_A(i, j):
        return i * m + j
        
    def idx_B(j, k):
        return j * p + k
        
    def idx_C(i, k):
        return k * n + i

    # Strassen coefficients for top level (4x4 * 4x6)
    # A blocks: 2x2
    # B blocks: 2x3
    # M_s = A_s B_s
    # A_s, B_s are linear combos of blocks.
    
    # Block definitions:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-2
    # B12: rows 0-1, cols 3-5
    # B21: rows 2-3, cols 0-2
    # B22: rows 2-3, cols 3-5
    
    # C blocks: 2x3
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    # Strassen products for 2x2 * 2x3
    # s=1: A1 = A11+A22, B1 = B11+B22
    # s=2: A2 = A21+A22, B2 = B11
    # s=3: A3 = A11, B3 = B12-B22
    # s=4: A4 = A22, B4 = B21-B11
    # s=5: A5 = A11+A12, B5 = B22
    # s=6: A6 = A21-A11, B6 = B11+B12
    # s=7: A7 = A12-A22, B7 = B21+B22
    
    strassen_defs = [
        # (A_coeff_dict, B_coeff_dict, C_coeff_dict)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # This assumes M_s = A_s B_s.
        
        # Let's verify the C combination for each M_s.
        # M1 contributes to C11 (+1) and C22 (+1)
        # M2 contributes to C21 (+1) and C22 (-1)
        # M3 contributes to C12 (+1) and C22 (+1)
        # M4 contributes to C11 (+1) and C21 (+1)
        # M5 contributes to C11 (-1) and C12 (+1)
        # M6 contributes to C22 (+1)
        # M7 contributes to C11 (+1)
    ]
    
    s_definitions = [
        # s, A_blocks {name: coeff}, B_blocks {name: coeff}, C_blocks {name: coeff}
        (1, {"11": 1, "22": 1}, {"11": 1, "22": 1}, {"11": 1, "22": 1}),
        (2, {"21": 1, "22": 1}, {"11": 1}, {"21": 1, "22": -1}),
        (3, {"11": 1}, {"12": 1, "22": -1}, {"12": 1, "22": 1}),
        (4, {"22": 1}, {"21": 1, "11": -1}, {"11": 1, "21": 1}),
        (5, {"11": 1, "12": 1}, {"22": 1}, {"11": -1, "12": 1}),
        (6, {"21": 1, "11": -1}, {"11": 1, "12": 1}, {"22": 1}),
        (7, {"12": 1, "22": -1}, {"21": 1, "22": 1}, {"11": 1}),
    ]
    
    col = 0
    
    # Helper to get local indices
    def get_A_local_indices(block, i, j):
        # i, j are 0..1 for 2x2 block
        if block == "11":
            return idx_A(i, j)
        elif block == "12":
            return idx_A(i, j + 2)
        elif block == "21":
            return idx_A(i + 2, j)
        else:
            return idx_A(i + 2, j + 2)
            
    def get_B_local_indices(block, i, j):
        # i is 0..1 (row), j is 0..2 (col) for 2x3 block
        if block == "11":
            return idx_B(i, j)
        elif block == "12":
            return idx_B(i, j + 3)
        elif block == "21":
            return idx_B(i + 2, j)
        else:
            return idx_B(i + 2, j + 3)

    def get_C_local_indices(block, i, j):
        # i is 0..1 (row), j is 0..2 (col) for 2x3 block
        if block == "11":
            return idx_C(i, j)
        elif block == "12":
            return idx_C(i, j + 3)
        elif block == "21":
            return idx_C(i + 2, j)
        else:
            return idx_C(i + 2, j + 3)

    # Function to add a term defined by U_coeffs, V_coeffs, W_coeffs
    # U_coeffs: dict global_A_idx -> coeff
    # V_coeffs: dict global_B_idx -> coeff
    # W_coeffs: dict global_C_idx -> coeff
    def add_term(U_coeffs, V_coeffs, W_coeffs):
        nonlocal col
        for idx, val in U_coeffs.items():
            if val != 0:
                U[idx, col] += val
        for idx, val in V_coeffs.items():
            if val != 0:
                V[idx, col] += val
        for idx, val in W_coeffs.items():
            if val != 0:
                W[idx, col] += val
        col += 1

    # Decompose M_{2,2,3} into 11 terms
    # A_local is 2x2, B_local is 2x3
    # We use 7 Strassen terms for A * B[:, :2]
    # And 4 schoolbook terms for A * B[:, 2]
    
    # Let A_L = [[a, b], [c, d]]
    # Let B_L = [[g, h, k], [i, j, l]]
    # B_L_left = [[g, h], [i, j]]
    # B_L_right = [[k], [l]]
    
    # Strassen for A_L * B_L_left:
    # S1 = (a+d)(g+j)
    # S2 = (c+d)g
    # S3 = a(h-j)
    # S4 = d(i-g)
    # S5 = (a+b)j
    # S6 = (c-a)(g+h)
    # S7 = (b-d)(i+j)
    
    # C_L_left = [[C11, C12], [C21, C22]]
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Schoolbook for A_L * B_L_right:
    # S8 = a k
    # S9 = b l
    # S10 = c k
    # S11 = d l
    # C13 = S8 + S9
    # C23 = S10 + S11
    
    # We need to map these to global indices.
    
    for s, a_blocks, b_blocks, c_blocks in s_definitions:
        # Construct local A_L and B_L as linear combos of global A and B
        # A_L[i][j] = sum c_a * A_block(i,j)
        # B_L[i][k] = sum d_b * B_block(i,k)
        
        A_L = [[{} for _ in range(2)] for _ in range(2)]
        B_L = [[{} for _ in range(3)] for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                for block_name, coeff in a_blocks.items():
                    g_idx = get_A_local_indices(block_name, i, j)
                    A_L[i][j][g_idx] = A_L[i][j].get(g_idx, 0) + coeff
                    
        for i in range(2):
            for k in range(3):
                for block_name, coeff in b_blocks.items():
                    g_idx = get_B_local_indices(block_name, i, k)
                    B_L[i][k][g_idx] = B_L[i][k].get(g_idx, 0) + coeff
        
        # Extract A_L entries
        a = A_L[0][0]
        b = A_L[0][1]
        c = A_L[1][0]
        d = A_L[1][1]
        
        # Extract B_L entries
        g = B_L[0][0]
        h = B_L[0][1]
        k_col = B_L[0][2]
        i_col = B_L[1][0]
        j_col = B_L[1][1]
        l_col = B_L[1][2]
        
        # Helper to add linear combo
        def combine_dicts(*dicts):
            res = {}
            for d in dicts:
                for key, val in d.items():
                    res[key] = res.get(key, 0) + val
            return res
            
        def subtract_dicts(d1, d2):
            res = {}
            for key, val in d1.items():
                res[key] = res.get(key, 0) + val
            for key, val in d2.items():
                res[key] = res.get(key, 0) - val
            return res

        # Define W targets for each local C entry
        # We need to add contributions to global C.
        # W_coeffs for a term is sum of coeff * C_block_indices
        
        def get_W_targets(local_i, local_j, coeff):
            # local_i, local_j are indices in 2x3 C block
            targets = {}
            for block_name, c_coeff in c_blocks.items():
                g_idx = get_C_local_indices(block_name, local_i, local_j)
                targets[g_idx] = targets.get(g_idx, 0) + coeff * c_coeff
            return targets

        # S1 = (a+d)(g+j_col)
        u_s1 = combine_dicts(a, d)
        v_s1 = combine_dicts(g, j_col)
        w_s1 = combine_dicts(get_W_targets(0, 0, 1), get_W_targets(1, 1, 1))
        add_term(u_s1, v_s1, w_s1)
        
        # S2 = (c+d)g
        u_s2 = combine_dicts(c, d)
        v_s2 = g
        w_s2 = combine_dicts(get_W_targets(1, 0, 1), get_W_targets(1, 1, -1))
        add_term(u_s2, v_s2, w_s2)
        
        # S3 = a(h-j_col)
        u_s3 = a
        v_s3 = subtract_dicts(h, j_col)
        w_s3 = combine_dicts(get_W_targets(0, 1, 1), get_W_targets(1, 1, 1))
        add_term(u_s3, v_s3, w_s3)
        
        # S4 = d(i_col-g)
        u_s4 = d
        v_s4 = subtract_dicts(i_col, g)
        w_s4 = combine_dicts(get_W_targets(0, 0, 1), get_W_targets(1, 0, 1))
        add_term(u_s4, v_s4, w_s4)
        
        # S5 = (a+b)j_col
        u_s5 = combine_dicts(a, b)
        v_s5 = j_col
        w_s5 = combine_dicts(get_W_targets(0, 0, -1), get_W_targets(0, 1, 1))
        add_term(u_s5, v_s5, w_s5)
        
        # S6 = (c-a)(g+h)
        u_s6 = subtract_dicts(c, a)
        v_s6 = combine_dicts(g, h)
        w_s6 = get_W_targets(1, 1, 1)
        add_term(u_s6, v_s6, w_s6)
        
        # S7 = (b-d)(i_col+j_col)
        u_s7 = subtract_dicts(b, d)
        v_s7 = combine_dicts(i_col, j_col)
        w_s7 = get_W_targets(0, 0, 1)
        add_term(u_s7, v_s7, w_s7)
        
        # S8 = a k_col
        u_s8 = a
        v_s8 = k_col
        w_s8 = get_W_targets(0, 2, 1)
        add_term(u_s8, v_s8, w_s8)
        
        # S9 = b l_col
        u_s9 = b
        v_s9 = l_col
        w_s9 = get_W_targets(0, 2, 1)
        add_term(u_s9, v_s9, w_s9)
        
        # S10 = c k_col
        u_s10 = c
        v_s10 = k_col
        w_s10 = get_W_targets(1, 2, 1)
        add_term(u_s10, v_s10, w_s10)
        
        # S11 = d l_col
        u_s11 = d
        v_s11 = l_col
        w_s11 = get_W_targets(1, 2, 1)
        add_term(u_s11, v_s11, w_s11)

    return {"U": U, "V": V, "W": W}
