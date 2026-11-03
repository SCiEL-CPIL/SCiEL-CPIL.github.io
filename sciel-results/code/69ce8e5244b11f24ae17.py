import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # BCLR rank-14 4x4x4 decomposition
    # x1=A11, x2=A21+A22, x3=A11-A21, x4=A22, x5=A12+A22, x6=A12-A22, x7=A11+A12
    # y1=B11, y2=B11-B21, y3=B22, y4=B12, y5=B11+B21, y6=B21-B22, y7=B11+B22
    # 7 sums x 4 terms each = rank 28 per 4x4x4 block
    
    # Each term (u, v, w) represents: A_row_u * B_col_v contributes to C with w
    # u: A row (0-3), v: B column (0-3), w: list of (C_row, C_col, coeff)
    
    terms = [
        # z1 = x1 * y1 = A11 * B11
        (0, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z2 = x2 * y2 = (A21+A22) * (B11-B21)
        (1, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z3 = x3 * y3 = (A11-A21) * B22
        (0, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z4 = x4 * y4 = A22 * B12
        (2, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z5 = x5 * y5 = (A12+A22) * (B11+B21)
        (1, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z6 = x6 * y6 = (A12-A22) * (B21-B22)
        (1, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 1, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (2, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        # z7 = x7 * y7 = (A11+A12) * (B11+B22)
        (0, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 0, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (0, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
        (1, 2, [(0, 0, 1), (0, 1, -1), (1, 1, -1), (1, 0, 1)]),
    ]
    
    rank = 56  # 28 per half * 2 halves
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for half in range(2):
        for t_idx, (u_idx, v_idx, w_list) in enumerate(terms):
            col = half * 28 + t_idx
            
            # U: A row u_idx
            U[u_idx, col] = 1.0
            
            # V: B column v_idx, offset by half*4
            V[v_idx + half * 4, col] = 1.0
            
            # W: C contributions, offset by half*4
            for (c_row, c_col, coeff) in w_list:
                W[(c_col + half * 4) * 4 + c_row, col] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
