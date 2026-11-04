import numpy as np

DIMENSIONS = [4, 4, 8]

def get_strassen_2x2():
    # U2, V2, W2 for 2x2x2 Strassen
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    return U2, V2, W2

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 using Strassen
    U2, V2, W2 = get_strassen_2x2()
    
    # Outer Strassen factors (same as inner for 2x2 blocks)
    U_out = U2
    V_out = V2
    W_out = W2
    
    # Build 4x4x4 factors
    rank_4 = 49
    U4 = np.zeros((16, rank_4))
    V4 = np.zeros((16, rank_4))
    W4 = np.zeros((16, rank_4))
    
    for k in range(7):
        for l in range(7):
            c = 7 * k + l
            # U contribution
            for b_A in range(4):
                coeff_u = U_out[k, b_A]
                if coeff_u != 0:
                    # Block b_A corresponds to rows/cols
                    # b_A=0: A00, b_A=1: A01, b_A=2: A10, b_A=3: A11
                    # Row/Col index in 4x4:
                    r_block = b_A // 2
                    c_block = b_A % 2
                    for i in range(2):
                        for j in range(2):
                            row_idx = r_block * 2 + i
                            col_idx = c_block * 2 + j
                            U4[row_idx * 4 + col_idx, c] += coeff_u * U2[i * 2 + j, l]
            
            # V contribution
            for b_B in range(4):
                coeff_v = V_out[k, b_B]
                if coeff_v != 0:
                    r_block = b_B // 2
                    c_block = b_B % 2
                    for i in range(2):
                        for j in range(2):
                            row_idx = r_block * 2 + i
                            col_idx = c_block * 2 + j
                            V4[row_idx * 4 + col_idx, c] += coeff_v * V2[i * 2 + j, l]
            
            # W contribution
            # W_out[k, b_C] where b_C is the block index in C
            # C = P1...P7 combination
            for b_C in range(4):
                coeff_w = W_out[k, b_C]
                if coeff_w != 0:
                    r_block = b_C // 2
                    c_block = b_C % 2
                    for i in range(2):
                        for j in range(2):
                            row_idx = r_block * 2 + i
                            col_idx = c_block * 2 + j
                            # W index: k*n + i? No, W is for C.
                            # C is 4x4. Index (i, k) in C.
                            # The problem says W[k*n+i]. Here n=4.
                            # So for C4, index is col*4 + row.
                            W4[col_idx * 4 + row_idx, c] += coeff_w * W2[i * 2 + j, l]

    # Combine for 4x4x8
    rank_total = 98
    U = np.zeros((16, rank_total))
    V = np.zeros((32, rank_total))
    W = np.zeros((32, rank_total))
    
    # First 49 columns
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Next 49 columns
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
