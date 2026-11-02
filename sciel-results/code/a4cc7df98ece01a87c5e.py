import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(n * p)]
    
    # Define block indices
    # A is 4x4
    # Block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
    def get_A_indices(br, bc):
        idxs = []
        for i in range(2):
            for j in range(2):
                r = 2 * br + i
                c = 2 * bc + j
                idxs.append(r * 4 + c)
        return idxs
        
    # B is 4x6
    # Block (br, bc) has rows 2*br..2*br+1, cols 3*bc..3*bc+2
    def get_B_indices(br, bc):
        idxs = []
        for i in range(2):
            for j in range(3):
                r = 2 * br + i
                c = 3 * bc + j
                idxs.append(r * 6 + c)
        return idxs

    # Precompute block indices
    A_blocks = {}
    for br in range(2):
        for bc in range(2):
            A_blocks[(br, bc)] = get_A_indices(br, bc)
            
    B_blocks = {}
    for br in range(2):
        for bc in range(2):
            B_blocks[(br, bc)] = get_B_indices(br, bc)

    # Helper to create a combination of blocks
    # combos: list of (br, bc, coeff)
    # Returns list of (global_idx, coeff) for each element in the 2x2 or 2x3 matrix
    def make_X_combos(combos):
        # X is 2x2
        # Returns list of 4 lists. Each list is [(idx, coeff), ...]
        res = [[], [], [], []]
        for br, bc, coeff in combos:
            idxs = A_blocks[(br, bc)]
            # idxs[0] is (0,0), idxs[1] is (0,1), idxs[2] is (1,0), idxs[3] is (1,1)
            for k in range(4):
                res[k].append((idxs[k], coeff))
        return res

    def make_Y_combos(combos):
        # Y is 2x3
        # Returns list of 6 lists. Each list is [(idx, coeff), ...]
        res = [[], [], [], [], [], []]
        for br, bc, coeff in combos:
            idxs = B_blocks[(br, bc)]
            # idxs[0..2] are row 0, idxs[3..5] are row 1
            for k in range(6):
                res[k].append((idxs[k], coeff))
        return res

    # Define the 7 Strassen steps
    # Each step: (X_combos, Y_combos, C_coeffs)
    # C_coeffs: {(br, bc): coeff}
    
    steps = []
    
    # 1. M1 = (A00 + A11)(B00 + B11)
    steps.append((
        make_X_combos([(0,0,1), (1,1,1)]),
        make_Y_combos([(0,0,1), (1,1,1)]),
        {(0,0): 1, (1,1): 1} # C00 += M1, C11 += M1
    ))
    
    # 2. M2 = (A10 + A11)B11
    steps.append((
        make_X_combos([(1,0,1), (1,1,1)]),
        make_Y_combos([(1,1,1)]),
        {(1,0): 1, (1,1): -1} # C10 += M2, C11 -= M2
    ))
    
    # 3. M3 = A00(B01 - B11)
    steps.append((
        make_X_combos([(0,0,1)]),
        make_Y_combos([(0,1,1), (1,1,-1)]),
        {(0,1): 1, (1,1): 1} # C01 += M3, C11 += M3
    ))
    
    # 4. M4 = (A11 - A00)(B00 + B10)
    steps.append((
        make_X_combos([(1,1,1), (0,0,-1)]),
        make_Y_combos([(0,0,1), (1,0,1)]),
        {(0,0): 1, (1,1): 1} # C00 += M4, C11 += M4
    ))
    
    # 5. M5 = (A00 + A01)B10
    steps.append((
        make_X_combos([(0,0,1), (0,1,1)]),
        make_Y_combos([(1,0,1)]),
        {(0,0): -1, (0,1): 1} # C00 -= M5, C01 += M5
    ))
    
    # 6. M6 = (A01 - A11)(B00 + B01)
    steps.append((
        make_X_combos([(0,1,1), (1,1,-1)]),
        make_Y_combos([(0,0,1), (0,1,1)]),
        {(1,0): 1, (0,1): 1} # C10 += M6, C01 += M6
    ))
    
    # 7. M7 = (A10 - A00)(B01 + B11)
    steps.append((
        make_X_combos([(1,0,1), (0,0,-1)]),
        make_Y_combos([(0,1,1), (1,1,1)]),
        {(0,0): 1, (1,1): 1} # C00 += M7, C11 += M7
    ))
    
    col = 0
    for x_combos, y_combos, c_coeffs in steps:
        # x_combos has 4 entries (for X_00, X_01, X_10, X_11)
        # y_combos has 6 entries (for Y_00, Y_01, Y_02, Y_10, Y_11, Y_12)
        
        for a in range(2): # Row of X (and Z)
            for b in range(2): # Col of X / Row of Y
                for c in range(3): # Col of Y (and Z)
                    
                    # U part: X_ab
                    u_idx_in_block = a * 2 + b
                    for idx, coeff in x_combos[u_idx_in_block]:
                        U[idx][col] += coeff
                        
                    # V part: Y_bc
                    v_idx_in_block = b * 3 + c
                    for idx, coeff in y_combos[v_idx_in_block]:
                        V[idx][col] += coeff
                        
                    # W part: Z_ac contributes to C blocks
                    for (br, bc), coeff_c in c_coeffs.items():
                        if coeff_c != 0:
                            # Global row: 2*br + a
                            # Global col: 3*bc + c
                            global_row = 2 * br + a
                            global_col = 3 * bc + c
                            w_idx = global_row * 6 + global_col
                            W[w_idx][col] += coeff_c
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
