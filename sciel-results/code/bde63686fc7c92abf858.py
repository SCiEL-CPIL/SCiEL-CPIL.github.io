import time
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-84 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses 3 independent 4x4x2 subproblems, each decomposed into 4 rank-7 Strassen 2x2x2 products.
    """
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U: (4, 7)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 0, 0, 1],   # a01
        [0, 1, 0, 0, 1, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ]
    
    # V: (4, 7)
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ]
    
    # W: (4, 7) - Trace convention k*2+i
    # c00: m1 + m4 - m5 + m7
    # c01: m3 + m5
    # c10: m2 + m4
    # c11: m1 - m2 + m3 + m6
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # c00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # c01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],   # c10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # c11 (k=1, i=1)
    ]
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over 3 chunks of p (size 2)
    for t in range(3):
        # Iterate over 4 blocks of A (2x2)
        for I in range(2):
            for J in range(2):
                # Base column index for this Strassen instance
                base_col = t * 28 + (I * 4 + J) * 7
                
                # Iterate over 7 Strassen terms
                for r in range(7):
                    col = base_col + r
                    
                    # U contributions
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = U_strassen[i_loc * 2 + j_loc][r]
                            if val != 0:
                                row_u = (2 * I + i_loc) * m + (2 * J + j_loc)
                                U[row_u][col] += val
                                
                    # V contributions
                    # B block is determined by J (0 for top, 1 for bottom)
                    K_block = J
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = V_strassen[j_loc * 2 + k_loc][r]
                            if val != 0:
                                row_v = (2 * K_block + j_loc) * p + (2 * t + k_loc)
                                V[row_v][col] += val
                                
                    # W contributions
                    # C block is determined by I (0 for top, 1 for bottom)
                    I_out = I
                    for k_loc in range(2):
                        for i_loc in range(2):
                            val = W_strassen[k_loc * 2 + i_loc][r]
                            if val != 0:
                                row_w = (2 * t + k_loc) * n + (2 * I_out + i_loc)
                                W[row_w][col] += val
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
