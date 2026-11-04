import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps
    # alpha, beta, gamma are dicts {(i,j): coeff}
    strassen_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 (cols 0-3 of B, 0-3 of C)
    # Rank 49
    for s1 in range(7):
        a1, b1, c1 = strassen_222[s1]
        for s2 in range(7):
            a2, b2, c2 = strassen_222[s2]
            col = s1 * 7 + s2
            
            # U (A)
            for (bi, bj), va1 in a1.items():
                for (si, sj), va2 in a2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += va1 * va2
                    
            # V (B)
            for (bi, bj), vb1 in b1.items():
                for (si, sj), vb2 in b2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += vb1 * vb2
                    
            # W (C)
            for (bi, bj), vc1 in c1.items():
                for (si, sj), vc2 in c2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += vc1 * vc2

    # Part 2: 4x4x2 (cols 4-5 of B, 4-5 of C)
    # Rank 28
    # C_top = A00 B_top + A01 B_bot
    # C_bot = A10 B_top + A11 B_bot
    # A blocks: 00, 01, 10, 11
    # B blocks: top (rows 0-1), bot (rows 2-3)
    # C blocks: top (rows 0-1), bot (rows 2-3)
    
    # Multiplications:
    # 1. A00 * B_top -> C_top
    # 2. A01 * B_bot -> C_top
    # 3. A10 * B_top -> C_bot
    # 4. A11 * B_bot -> C_bot
    
    mults = [
        ((0, 0), 0, 0), # A_block, B_block_idx (0=top, 1=bot), C_block_idx
        ((0, 1), 1, 0),
        ((1, 0), 0, 1),
        ((1, 1), 1, 1)
    ]
    
    col_start = 49
    for m_idx, (a_blk, b_blk_idx, c_blk_idx) in enumerate(mults):
        for s in range(7):
            a_s, b_s, c_s = strassen_222[s]
            col = col_start + m_idx * 7 + s
            
            # U
            for (si, sj), va in a_s.items():
                r = (a_blk[0] * 2 + si) * m + (a_blk[1] * 2 + sj)
                U[r][col] += va
                
            # V
            # B block row offset: 0 for top, 2 for bot
            b_row_off = b_blk_idx * 2
            for (si, sj), vb in b_s.items():
                r = (b_row_off + si) * p + (4 + sj)
                V[r][col] += vb
                
            # W
            # C block row offset: 0 for top, 2 for bot
            c_row_off = c_blk_idx * 2
            for (si, sj), vc in c_s.items():
                r = (4 + sj) * n + (c_row_off + si)
                W[r][col] += vc

    return {"U": U, "V": V, "W": W}
