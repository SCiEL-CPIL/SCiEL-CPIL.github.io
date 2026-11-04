import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps: (alpha, beta, gamma)
    # Each is a dict mapping (i,j) to coefficient
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4 x 4x4 -> 4x4 (columns 0-3 of B, columns 0-3 of C)
    # Rank 49 = 7 * 7
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_steps_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = s * 7 + s2
            
            # U: A is 4x4, row = i*m + j
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2
            
            # V: B1 is 4x4 (cols 0-3 of B), row = i*p + j
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2
            
            # W: C1 is 4x4 (cols 0-3 of C), index = col_idx * n + row_idx
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2
    
    # Part 2: 4x4 x 4x2 -> 4x2 (columns 4-5 of B, columns 4-5 of C)
    # Four 2x2x2 multiplications, each rank 7, total 28
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    mults = [
        ((0, 0), 0),  # A00 (bi=0,bj=0) x B_top (b_row=0) -> C_top (c_row=0)
        ((0, 1), 1),  # A01 (bi=0,bj=1) x B_bot (b_row=1) -> C_top (c_row=0)
        ((1, 0), 0),  # A10 (bi=1,bj=0) x B_top (b_row=0) -> C_bot (c_row=1)
        ((1, 1), 1),  # A11 (bi=1,bj=1) x B_bot (b_row=1) -> C_bot (c_row=1)
    ]
    
    for m_idx, (a_block, b_half) in enumerate(mults):
        a_bi, a_bj = a_block
        c_half = a_bi  # C row half matches A row half
        
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_steps_222[s2]
            col = 49 + m_idx * 7 + s2
            
            # U: A block at (a_bi, a_bj)
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_bi * 2 + si) * m + (a_bj * 2 + sj)
                U[r][col] += val_a2
            
            # V: B2 block, rows (b_half*2 + si), cols (4 + sj)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_half * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2
            
            # W: C2 block, col = 4 + sj, row = c_half*2 + si
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_half * 2 + si)
                W[r][col] += val_c2
    
    return {"U": U, "V": V, "W": W}
