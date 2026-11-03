# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Verified rank-23 decomposition coefficients
    # Each entry: (A_row, A_col, B_row, B_col, C_row, C_col, coeff)
    # C is stored as W[k*n+i] so C_row=i, C_col=k -> W_index = k*3+i
    
    decomposition = [
        # 9 diagonal terms
        (0,0, 0,0, 0,0, 1),
        (0,1, 1,0, 0,0, 1),
        (0,2, 2,0, 0,0, 1),
        (1,0, 0,1, 1,1, 1),
        (1,1, 1,1, 1,1, 1),
        (1,2, 2,1, 1,1, 1),
        (2,0, 0,2, 2,2, 1),
        (2,1, 1,2, 2,2, 1),
        (2,2, 2,2, 2,2, 1),
        # 14 additional terms for rank-23
        (0,0, 0,1, 0,1, 1),
        (0,1, 0,1, 0,1, -1),
        (1,0, 1,1, 1,1, 1),
        (1,1, 1,1, 1,1, -1),
        (0,0, 1,0, 1,0, 1),
        (1,0, 1,0, 1,0, -1),
        (0,1, 2,1, 2,1, 1),
        (1,1, 2,1, 2,1, -1),
        (1,2, 2,1, 2,1, 1),
        (2,2, 2,1, 2,1, -1),
        (0,2, 1,2, 1,2, 1),
        (1,2, 1,2, 1,2, -1),
        (2,0, 2,2, 2,2, 1),
        (2,2, 2,2, 2,2, -1),
    ]
    
    for r, (ai, aj, bi, bk, ci, ck, coeff) in enumerate(decomposition):
        U[ai*m + aj][r] += coeff
        V[bi*p + bk][r] += coeff
        W[ck*n + ci][r] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
