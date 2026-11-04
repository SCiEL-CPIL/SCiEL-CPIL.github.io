import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    n, m, p = 4, 4, 6
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps: (alpha, beta, gamma)
    # Each is a dict from (i,j) to coefficient
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B, columns 0-3 of C)
    col_offset = 0
    for s_outer in range(7):
        if time.time() > deadline - 10:
            break
        alpha_o, beta_o, gamma_o = strassen_steps[s_outer]
        for s_inner in range(7):
            alpha_i, beta_i, gamma_i = strassen_steps[s_inner]
            col = col_offset + s_outer * 7 + s_inner
            
            # U: A is 4x4
            for (bi, bj), val_o in alpha_o.items():
                for (si, sj), val_i in alpha_i.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_o * val_i
            
            # V: B1 is 4x4 (columns 0-3 of B)
            for (bi, bj), val_o in beta_o.items():
                for (si, sj), val_i in beta_i.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_o * val_i
            
            # W: C1 is 4x4 (columns 0-3 of C)
            # W index: col * n + row
            for (bi, bj), val_o in gamma_o.items():
                for (si, sj), val_i in gamma_i.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_o * val_i
    
    # Part 2: 4x4 * 4x2 (columns 4-5 of B, columns 4-5 of C)
    col_offset = 49
    mults = [
        ((0, 0), (0, 0), (0, 0)),  # A00 * B0 -> C0
        ((0, 1), (1, 0), (0, 0)),  # A01 * B1 -> C0
        ((1, 0), (0, 0), (1, 0)),  # A10 * B0 -> C1
        ((1, 1), (1, 0), (1, 0)),  # A11 * B1 -> C1
    ]
    
    for m_idx, (a_block, b_block, c_block) in enumerate(mults):
        if time.time() > deadline - 10:
            break
        for s_inner in range(7):
            alpha_i, beta_i, gamma_i = strassen_steps[s_inner]
            col = col_offset + m_idx * 7 + s_inner
            
            # U: A block
            for (si, sj), val_i in alpha_i.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r][col] += val_i
            
            # V: B block (cols 4-5)
            for (si, sj), val_i in beta_i.items():
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r][col] += val_i
            
            # W: C block (cols 4-5)
            for (si, sj), val_i in gamma_i.items():
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r][col] += val_i
    
    return {"U": U, "V": V, "W": W}
