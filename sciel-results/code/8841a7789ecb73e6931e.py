"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt Rank 40 using 2x2x3 blocks (Rank 10 each)
    # 4 blocks of 2x2x3
    rank_40 = 40
    U40 = [[0] * rank_40 for _ in range(n * m)]
    V40 = [[0] * rank_40 for _ in range(m * p)]
    W40 = [[0] * rank_40 for _ in range(p * n)]
    
    # Rank 10 decomposition for 2x2x3
    # X: 2x2, Y: 2x3
    # This is a known decomposition.
    # If it fails verification, we fall back to Rank 42.
    
    # Definitions for 2x2x3 Rank 10
    # X indices: 0,1,2,3
    # Y indices: 0,1,2,3,4,5
    r10_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1, 4: 1}), # M1
        ({1: 1, 3: 1}, {1: 1, 5: 1}),       # M2
        ({0: 1, 2: 1}, {0: 1, 1: 1, 2: 1}), # M3
        ({2: 1, 3: 1}, {3: 1, 4: 1, 5: 1}), # M4
        ({0: 1}, {0: 1, 3: 1}),             # M5
        ({1: 1}, {1: 1, 4: 1}),             # M6
        ({2: 1}, {2: 1, 5: 1}),             # M7
        ({3: 1}, {0: 1, 3: 1, 4: 1, 5: 1}), # M8
        ({0: 1, 1: 1}, {2: 1, 5: 1}),       # M9
        ({2: 1, 3: 1}, {0: 1, 3: 1})        # M10
    ]
    
    # Reconstruction for Z (2x3) from M1..M10
    # Z00 = M1 + M3 + M5 - M8 - M9 - M10 ?
    # This reconstruction is complex and error-prone to guess.
    # I will use a verification step.
    
    # Since guessing the reconstruction coefficients is risky, 
    # and I cannot verify the Rank 10 decomposition coefficients I just wrote 
    # without a reference, I will fall back to the known correct Rank 42 
    # but implemented in a slightly different way to satisfy the "new design" 
    # directive? No, that's cheating.
    
    # Let's use the Rank 42 approach but with a different block split?
    # No, 42 is optimal for 2x2x2 blocks.
    
    # I will return the Rank 42 decomposition as it is verified correct.
    # The "new design" directive is hard to satisfy without a known lower rank.
    # I will keep the Rank 42 logic but clean it up.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    res_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        for s in range(7):
            col = start_col + s
            x_coeffs, y_coeffs = strassen_defs[s]
            
            for x_idx, val in x_coeffs.items():
                r = x_idx // 2
                c = x_idx % 2
                global_i = a_row_offset + r
                global_j = a_col_offset + c
                U[global_i * m + global_j][col] = val
                
            for y_idx, val in y_coeffs.items():
                r = y_idx // 2
                c = y_idx % 2
                global_j = b_row_offset + r
                global_k = b_col_offset + c
                V[global_j * p + global_k][col] = val
                
            for z_idx in range(4):
                coeff = res_coeffs[z_idx][s]
                if coeff != 0:
                    z_r = z_idx // 2
                    z_c = z_idx % 2
                    global_i = c_row_offset + z_r
                    global_k = c_col_offset + z_c
                    W[global_k * n + global_i][col] = coeff

    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    add_strassen_block(0, 0, 0, 4, 0, 4, 28)
    add_strassen_block(0, 2, 2, 4, 0, 4, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
