"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank for 4x4x4 is 49 (2x2 Strassen blocks)
    # Total rank is 2 * 49 = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms for 2x2 multiplication
    # Each term is (U_dict, V_dict, W_dict)
    # Keys are (row, col)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # m1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # m2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}), # m3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,0): 1}), # m4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}), # m5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}), # m6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1}) # m7
    ]
    
    # Define Block Strassen combinations for 4x4
    # Each entry is (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    block_strassen_4x4 = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (1,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,1,1)])
    ]
    
    # Process two halves of the 4x4x8 multiplication
    for h in range(2):
        # h=0: B cols 0-3, C cols 0-3
        # h=1: B cols 4-7, C cols 4-7
        b_col_offset = 4 * h
        c_col_offset = 4 * h
        
        for l in range(7):
            A_blocks, B_blocks, C_blocks = block_strassen_4x4[l]
            
            for s in range(7):
                col = h * 49 + l * 7 + s
                U_dict, V_dict, W_dict = strassen_2x2[s]
                
                # Process U
                for (r, c), coeff in U_dict.items():
                    for (br, bc, a_coeff) in A_blocks:
                        global_r = 2 * br + r
                        global_c = 2 * bc + c
                        U[global_r * m + global_c][col] += coeff * a_coeff
                        
                # Process V
                for (r, c), coeff in V_dict.items():
                    for (br, bc, b_coeff) in B_blocks:
                        global_r = 2 * br + r
                        global_c = 2 * bc + c + b_col_offset
                        V[global_r * p + global_c][col] += coeff * b_coeff
                        
                # Process W
                for (r, c), coeff in W_dict.items():
                    for (br, bc, c_coeff) in C_blocks:
                        global_r = 2 * br + r
                        global_c = 2 * bc + c + c_col_offset
                        # W index is k*n + i. Here global_r is k, global_c is i.
                        W[global_r * n + global_c][col] += coeff * c_coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
