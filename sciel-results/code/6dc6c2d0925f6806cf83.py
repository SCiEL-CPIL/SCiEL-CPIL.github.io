"""Strassen-based rank-28 decomposition for 2x2 by 2x8 matrix multiplication."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    p_blocks = p // m  # 4 blocks of 2 columns each
    
    # Strassen's 7 products for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # M1 = (a+d)(e+h)  -> C00 += 1, C11 += 1
    # M2 = (c+d)e      -> C10 += 1, C11 -= 1
    # M3 = a(f-h)      -> C01 += 1, C11 += 1
    # M4 = d(g-f)      -> C00 += 1, C10 += 1
    # M5 = (a+b)h      -> C00 -= 1, C01 += 1
    # M6 = (c-d)(g+h)  -> C11 += 1
    # M7 = (a-b)(e+f)  -> C00 += 1, C01 -= 1
    
    # U entries for A coefficients (same for all blocks since A is shared)
    # A flat indices: 0:a, 1:b, 2:c, 3:d
    U_strassen = [
        [1, 0, 0, 0],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [0, 0, 1, -1], # M6: c-d
        [1, -1, 0, 0], # M7: a-b
    ]
    
    # V entries for B coefficients (per 2x2 block)
    # B flat indices within a block: 0:e, 1:f, 2:g, 3:h
    V_strassen = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, -1, 1, 0], # M4: g-f
        [0, 0, 0, 1],  # M5: h
        [0, 0, 1, 1],  # M6: g+h
        [1, 1, 0, 0],  # M7: e+f
    ]
    
    # W entries for C reconstruction coefficients (per 2x2 block)
    # C flat indices within a block: 0:C00, 1:C01, 2:C10, 3:C11
    W_strassen = [
        [1, 0, 0, 1],  # M1: C00 + 1, C11 + 1
        [0, 0, 1, -1], # M2: C10 + 1, C11 - 1
        [0, 1, 0, 1],  # M3: C01 + 1, C11 + 1
        [1, 0, 1, 0],  # M4: C00 + 1, C10 + 1
        [-1, 1, 0, 0], # M5: C00 - 1, C01 + 1
        [0, 0, 0, 1],  # M6: C11 + 1
        [1, -1, 0, 0], # M7: C00 + 1, C01 - 1
    ]
    
    rank = p_blocks * 7  # 4 * 7 = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    for block in range(p_blocks):
        for r in range(7):
            col = block * 7 + r
            # U: same for all blocks (A is shared)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U_strassen[r][i * m + j]
            # V: different per block
            for j in range(m):
                for k in range(m):
                    # B index in full matrix: j * p + block * m + k
                    V[j * p + block * m + k][col] = V_strassen[r][j * m + k]
            # W: different per block, trace convention W[k*n+i, r]
            for i in range(n):
                for k in range(m):
                    # C index in full matrix: k * n + i (trace convention)
                    # But C has shape n x p = 2 x 8, so column index is block*m+k
                    # W index: (block*m + k) * n + i
                    W[(block * m + k) * n + i][col] = W_strassen[r][i * m + k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
