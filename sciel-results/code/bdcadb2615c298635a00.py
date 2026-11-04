import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Rank is 49 + 28 = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 steps for 2x2x2
    # Each step is (alpha, beta, gamma) where each is a dict from (i,j) to coeff
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 part (cols 0-48)
    # A is 4x4, B1 is 4x4 (cols 0-3 of B), C1 is 4x4 (cols 0-3 of C)
    col_start_1 = 0
    
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            
            col = col_start_1 + s * 7 + s2
            
            # Process U (A)
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
                    
            # Process V (B1)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
                    
            # Process W (C1)
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # Part 2: 4x4x2 part (cols 49-76)
    # A is 4x4, B2 is 4x2 (cols 4-5 of B), C2 is 4x2 (cols 4-5 of C)
    col_start_2 = 49
    
    # 4 multiplications of 2x2x2:
    # M1: A00 * B0 -> C0
    # M2: A01 * B1 -> C0
    # M3: A10 * B0 -> C1
    # M4: A11 * B1 -> C1
    # where B0 = B2[0:2, :], B1 = B2[2:4, :]
    #       C0 = C2[0:2, :], C1 = C2[2:4, :]
    
    mults = [
        ((0,0), (0,0), (0,0)), # A00, B0 -> C0
        ((0,1), (1,0), (0,0)), # A01, B1 -> C0
        ((1,0), (0,0), (1,0)), # A10, B0 -> C1
        ((1,1), (1,0), (1,0))  # A11, B1 -> C1
    ]
    
    for m_idx, (a_block, b_block, c_block) in enumerate(mults):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # Process U (A)
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_a2
                
            # Process V (B2)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
                
            # Process W (C2)
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r][col] += val_c2

    return {"U": U, "V": V, "W": W}
