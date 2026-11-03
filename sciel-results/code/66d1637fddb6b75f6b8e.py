import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 189  # 7 * 27 from Strassen 2x2 blocks of 3x3 schoolbook
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    # Strassen 2x2 block decomposition with 3x3 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], each block 3x3
    # C = [[C00, C01], [C10, C11]]
    
    # U coefficients for each Strassen product
    U_coefs = [
        [(0, 0, 1), (1, 1, 1)],   # M0: A00 + A11
        [(1, 0, 1), (1, 1, 1)],   # M1: A10 + A11
        [(0, 0, 1)],               # M2: A00
        [(1, 1, 1)],               # M3: A11
        [(0, 0, 1), (0, 1, 1)],   # M4: A00 + A01
        [(1, 0, 1), (0, 0, -1)],  # M5: A10 - A00
        [(0, 1, 1), (1, 1, -1)],  # M6: A01 - A11
    ]
    
    # V coefficients for each Strassen product
    V_coefs = [
        [(0, 0, 1), (1, 1, 1)],   # M0: B00 + B11
        [(0, 0, 1)],               # M1: B00
        [(0, 1, 1), (1, 1, -1)],  # M2: B01 - B11
        [(1, 0, 1), (0, 0, -1)],  # M3: B10 - B00
        [(1, 1, 1)],               # M4: B11
        [(0, 0, 1), (0, 1, 1)],   # M5: B00 + B01
        [(1, 0, 1), (1, 1, 1)],   # M6: B10 + B11
    ]
    
    # W coefficients: which C blocks each Strassen product contributes to
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5 + M6
    W_coefs = [
        [(0, 0, 1), (1, 1, 1)],      # M0: C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],     # M1: C10 += M1, C11 -= M1
        [(0, 1, 1)],                 # M2: C01 += M2
        [(0, 0, -1), (1, 0, 1)],     # M3: C00 -= M3, C10 += M3
        [(0, 0, 1), (0, 1, 1)],      # M4: C00 += M4, C01 += M4
        [(1, 1, 1)],                 # M5: C11 += M5
        [(0, 0, 1), (1, 1, 1)],      # M6: C00 += M6, C11 += M6
    ]
    
    block_size = 3
    
    for t in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    if time.time() > deadline:
                        break
                    
                    col = t * 27 + (i_l * 9 + j_l * 3 + k_l)
                    
                    for rb, cb, coeff in U_coefs[t]:
                        i_glob = rb * block_size + i_l
                        j_glob = cb * block_size + j_l
                        U[i_glob * m + j_glob][col] += coeff
                    
                    for rb, cb, coeff in V_coefs[t]:
                        j_glob = rb * block_size + j_l
                        k_glob = cb * block_size + k_l
                        V[j_glob * p + k_glob][col] += coeff
                    
                    for rb, cb, coeff in W_coefs[t]:
                        i_glob = rb * block_size + i_l
                        k_glob = cb * block_size + k_l
                        W[k_glob * n + i_glob][col] += coeff
    
    return {"U": U, "V": V, "W": W}
