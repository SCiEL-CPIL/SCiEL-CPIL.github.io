import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # A blocks (4x4) -> 2x2 blocks
    # B blocks (4x6) -> 2x3 blocks
    # C blocks (4x6) -> 2x3 blocks
    
    # Offsets: (row_offset, col_offset)
    A_offsets = {
        '00': (0, 0),
        '01': (0, 2),
        '10': (2, 0),
        '11': (2, 2)
    }
    
    B_offsets = {
        '00': (0, 0),
        '01': (0, 3),
        '10': (2, 0),
        '11': (2, 3)
    }
    
    C_offsets = {
        '00': (0, 0),
        '01': (0, 3),
        '10': (2, 0),
        '11': (2, 3)
    }
    
    # Strassen steps
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict block -> coeff
    # B_coeffs: dict block -> coeff
    # C_coeffs: dict block -> coeff (for the resulting M_s)
    
    steps = [
        # M1
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}), # Note: C coeffs are for M_s, not final C
        # Wait, the C_coeffs in the step definition should represent which C blocks M_s contributes to?
        # No, M_s is an intermediate product.
        # The final C is a sum of M_s.
        # So we need to define the mapping from M_s to C blocks.
        
        # Let's restructure.
        # Step s produces M_s.
        # M_s is added to C blocks with specific coefficients.
        
        # M1 -> C00 (+1), C11 (+1)
        # M2 -> C10 (+1), C11 (-1)
        # M3 -> C01 (+1), C11 (+1)
        # M4 -> C00 (+1), C10 (+1)
        # M5 -> C00 (-1), C01 (+1)
        # M6 -> C11 (+1)
        # M7 -> C00 (+1)
        
        # Let's verify:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each step s, we have:
        # A_blocks, B_blocks defining M_s
        # C_blocks, C_coeffs defining how M_s adds to C
    ]
    
    steps_data = [
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
    
    col = 0
    for s in range(7):
        a_coeffs, b_coeffs, c_coeffs = steps_data[s]
        
        # For each local j (0,1) and k (0,1,2)
        for j in range(2):
            for k in range(3):
                # U contributions
                for a_block, val in a_coeffs.items():
                    row_off, col_off = A_offsets[a_block]
                    for i in range(2):
                        global_i = row_off + i
                        global_j = col_off + j
                        U[global_i * m + global_j][col] += val
                        
                # V contributions
                for b_block, val in b_coeffs.items():
                    row_off, col_off = B_offsets[b_block]
                    global_j = row_off + j
                    global_k = col_off + k
                    V[global_j * p + global_k][col] += val
                    
                # W contributions
                for c_block, val in c_coeffs.items():
                    row_off, col_off = C_offsets[c_block]
                    for i in range(2):
                        global_i = row_off + i
                        global_k = col_off + k
                        # W[k*n + i, r]
                        W[global_k * n + global_i][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
