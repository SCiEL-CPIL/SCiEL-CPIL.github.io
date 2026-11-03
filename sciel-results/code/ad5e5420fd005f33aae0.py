import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 11 = 77
    rank = 77
    
    # U is n*m x rank
    U = [[0] * rank for _ in range(n * m)]
    # V is m*p x rank
    V = [[0] * rank for _ in range(m * p)]
    # W is p*n x rank
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps for the top level
    # Each step is (alpha_dict, beta_dict, gamma_dict)
    # alpha: block -> coeff for A
    # beta: block -> coeff for B
    # gamma: block -> coeff for C
    # Blocks: 00, 01, 10, 11
    
    steps = [
        # M1
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        # M2
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        # M3
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        # M4
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        # M5
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        # M6
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        # M7
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})
    ]
    
    # Helper to get row index in U for A_block[i1, j1]
    def get_u_row(block, i1, j1):
        if block == '00':
            i, j = i1, j1
        elif block == '01':
            i, j = i1, j1 + 2
        elif block == '10':
            i, j = i1 + 2, j1
        elif block == '11':
            i, j = i1 + 2, j1 + 2
        else:
            raise ValueError("Invalid block")
        return i * m + j
        
    # Helper to get row index in V for B_block[j1, k1]
    def get_v_row(block, j1, k1):
        if block == '00':
            j, k = j1, k1
        elif block == '01':
            j, k = j1, k1 + 3
        elif block == '10':
            j, k = j1 + 2, k1
        elif block == '11':
            j, k = j1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return j * p + k
        
    # Helper to get row index in W for C_block[i1, k1]
    # Convention: W[k*n+i, r]
    def get_w_row(block, i1, k1):
        if block == '00':
            i, k = i1, k1
        elif block == '01':
            i, k = i1, k1 + 3
        elif block == '10':
            i, k = i1 + 2, k1
        elif block == '11':
            i, k = i1 + 2, k1 + 3
        else:
            raise ValueError("Invalid block")
        return k * n + i

    # Define Strassen sub-steps for 2x2x2
    # Each step is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i,j) of A_sub to coeff
    # v_coeffs: dict mapping (i,j) of B_sub to coeff
    # w_coeffs: dict mapping (i,j) of C_sub (2x2) to coeff
    strassen_sub_steps = [
        # S1: (a+d)(e+h) -> C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # S2: (c+d)e -> C10
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # S3: a(f-h) -> C01
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # S4: d(g-e) -> C00, C10
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}),
        # S5: (a+b)h -> C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # S6: (c-a)(e+f) -> C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # S7: (b-d)(g+h) -> C00, C01
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1})
    ]
    
    # Define Standard sub-steps for 2x2x1
    # B_sub is 2x1, so only col 0 of B_sub (which is col 2 of the full 2x3 B_sub)
    # A_sub is 2x2
    # Result is 2x1
    standard_sub_steps = [
        # T1: a * B02 -> C0
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # T2: b * B12 -> C0
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        # T3: c * B02 -> C1
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        # T4: d * B12 -> C1
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1})
    ]
    
    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        
        # Process 7 Strassen sub-steps
        for sub_s in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_sub_steps[sub_s]
            
            # Fill U
            for (i1, j1), val in u_coeffs.items():
                for b, aval in alpha.items():
                    r = get_u_row(b, i1, j1)
                    U[r][col] += val * aval
                    
            # Fill V
            for (i1, j1), val in v_coeffs.items():
                # v_coeffs are for B_sub (2x2), which corresponds to cols 0,1 of B_block
                # So k1 is 0 or 1
                for b, bval in beta.items():
                    r = get_v_row(b, i1, j1)
                    V[r][col] += val * bval
                    
            # Fill W
            for (i1, k1), val in w_coeffs.items():
                # w_coeffs are for C_sub (2x2), which corresponds to cols 0,1 of C_block
                # So k1 is 0 or 1
                for b, gval in gamma.items():
                    r = get_w_row(b, i1, k1)
                    W[r][col] += val * gval
                    
            col += 1
            
        # Process 4 Standard sub-steps
        for sub_s in range(4):
            u_coeffs, v_coeffs, w_coeffs = standard_sub_steps[sub_s]
            
            # Fill U
            for (i1, j1), val in u_coeffs.items():
                for b, aval in alpha.items():
                    r = get_u_row(b, i1, j1)
                    U[r][col] += val * aval
                    
            # Fill V
            for (i1, j1), val in v_coeffs.items():
                # v_coeffs are for B_sub (2x1), which corresponds to col 2 of B_block
                # So k1 is 2
                for b, bval in beta.items():
                    r = get_v_row(b, i1, 2)
                    V[r][col] += val * bval
                    
            # Fill W
            for (i1, k1), val in w_coeffs.items():
                # w_coeffs are for C_sub (2x1), which corresponds to col 2 of C_block
                # So k1 is 2
                for b, gval in gamma.items():
                    r = get_w_row(b, i1, 2)
                    W[r][col] += val * gval
                    
            col += 1

    return {"U": U, "V": V, "W": W}
