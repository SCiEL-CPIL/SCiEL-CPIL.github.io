import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # Each factor is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_x_idx: val}
    # v_coeffs: dict {local_y_idx: val}
    # w_coeffs: dict {local_z_idx: val}
    
    strassen_factors = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): 1, (0,1): 1})
    ]
    
    # Define blocks
    # Block 1: A_L (rows 0-1, cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C cols 0-1
    # Block 2: A_R (rows 0-1, cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C cols 0-1
    # Block 3: A_L (rows 0-1, cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C cols 2-3
    # Block 4: A_R (rows 0-1, cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C cols 2-3
    
    blocks = [
        {
            'a_row_offset': 0, 'a_col_offset': 0,
            'b_row_offset': 0, 'b_col_offset': 0,
            'c_row_offset': 0, 'c_col_offset': 0
        },
        {
            'a_row_offset': 0, 'a_col_offset': 2,
            'b_row_offset': 2, 'b_col_offset': 0,
            'c_row_offset': 0, 'c_col_offset': 0
        },
        {
            'a_row_offset': 0, 'a_col_offset': 0,
            'b_row_offset': 0, 'b_col_offset': 2,
            'c_row_offset': 0, 'c_col_offset': 2
        },
        {
            'a_row_offset': 0, 'a_col_offset': 2,
            'b_row_offset': 2, 'b_col_offset': 2,
            'c_row_offset': 0, 'c_col_offset': 2
        }
    ]
    
    col = 0
    for block in blocks:
        for u_c, v_c, w_c in strassen_factors:
            # Map local indices to global
            # U index: i * m + j
            # V index: j * p + k
            # W index: k * n + i
            
            for (li, lj), val in u_c.items():
                gi = block['a_row_offset'] + li
                gj = block['a_col_offset'] + lj
                U[gi * m + gj][col] += val
            
            for (lj, lk), val in v_c.items():
                gi = block['b_row_offset'] + lj
                gj = block['b_col_offset'] + lk
                V[gi * p + gj][col] += val
            
            for (lk, li), val in w_c.items():
                gi = block['c_row_offset'] + li
                gj = block['c_col_offset'] + lk
                W[gj * n + gi][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
