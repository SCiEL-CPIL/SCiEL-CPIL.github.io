import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_naive_block(a_off, b_off, c_off):
        nonlocal col
        
        # 2x2 multiplication:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # We decompose each term A_ij * B_kl into a rank-1 tensor.
        # Term 1: A00 * B00 -> C00
        # Term 2: A01 * B10 -> C00
        # Term 3: A00 * B01 -> C01
        # Term 4: A01 * B11 -> C01
        # Term 5: A10 * B00 -> C10
        # Term 6: A11 * B10 -> C10
        # Term 7: A10 * B01 -> C11
        # Term 8: A11 * B11 -> C11
        
        terms = [
            # (a_idx, b_idx, c_idx, coeff)
            (0, 0, 0, 1), # A00 B00 -> C00
            (1, 2, 0, 1), # A01 B10 -> C00
            (0, 1, 1, 1), # A00 B01 -> C01
            (1, 3, 1, 1), # A01 B11 -> C01
            (2, 0, 2, 1), # A10 B00 -> C10
            (3, 2, 2, 1), # A11 B10 -> C10
            (2, 1, 3, 1), # A10 B01 -> C11
            (3, 3, 3, 1)  # A11 B11 -> C11
        ]
        
        for a_idx, b_idx, c_idx, coeff in terms:
            U[a_off[a_idx]][col] = coeff
            V[b_off[b_idx]][col] = coeff
            W[c_off[c_idx]][col] = coeff
            col += 1

    # Define offsets
    # A is 2x4.
    a_off_1 = [0, 1, 4, 5]
    a_off_2 = [2, 3, 6, 7]
    
    # B is 4x4.
    b_off_11 = [0, 1, 4, 5]
    b_off_12 = [2, 3, 6, 7]
    b_off_21 = [8, 9, 12, 13]
    b_off_22 = [10, 11, 14, 15]
    
    # C is 2x4. W index k*n + i.
    # C[:, 0:2]
    c_off_11 = [0, 2, 1, 3] # C00, C01, C10, C11 -> W[0], W[2], W[1], W[3]
    # C[:, 2:4]
    c_off_12 = [4, 6, 5, 7] # C02, C03, C12, C13 -> W[4], W[6], W[5], W[7]
    
    add_naive_block(a_off_1, b_off_11, c_off_11)
    add_naive_block(a_off_1, b_off_12, c_off_12)
    add_naive_block(a_off_2, b_off_21, c_off_11)
    add_naive_block(a_off_2, b_off_22, c_off_12)
    
    return {"U": U, "V": V, "W": W}
