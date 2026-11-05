import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 85 decomposition:
    # 49 (4x4x4 Strassen) + 16 (4x4x4x1) + 16 (4x1x4) + 4 (4x1x1)
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2x2 Strassen factors
    # u, v, w are 4x7
    s_u = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ]
    s_v = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, -1, 1, 0, 1]
    ]
    s_w = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Helper to add 4x4x4 Strassen terms
    def add_4x4_strassen(a_blocks, b_blocks, c_blocks, start_col):
        # a_blocks: list of (br, bc, coeff)
        # b_blocks: list of (br, bc, coeff)
        # c_blocks: list of (br, bc, coeff)
        
        for inner_k in range(7):
            col_idx = start_col + inner_k
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            for a_br, a_bc, a_coeff in a_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj][col_idx] += a_coeff * u_val

            for b_br, b_bc, b_coeff in b_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk][col_idx] += b_coeff * v_val

            for c_br, c_bc, c_coeff in c_blocks:
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gk = 2 * c_bc + lc
                            gi = 2 * c_br + lr
                            W[gk * 4 + gi][col_idx] += c_coeff * w_val

    # 7 Strassen terms for 4x4x4
    terms = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A21+A22)B11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A11+A12)B22
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]

    col = 0
    for t in terms:
        add_4x4_strassen(t[0], t[1], t[2], col)
        col += 7
        
    # 4x4x4x1 part (A_L B_TR)
    # A_L is 4x4 (rows 0-3, cols 0-3)
    # B_TR is 4x1 (rows 0-3, col 4)
    # C_R is 4x1 (rows 0-3, col 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # 4x1x4 part (A_R B_BL)
    # A_R is 4x1 (rows 0-3, col 4)
    # B_BL is 1x4 (row 4, cols 0-3)
    # C_L is 4x4 (rows 0-3, cols 0-3)
    # C[i, k] = A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1

    # 4x1x1 part (A_R B_BR)
    # A_R is 4x1
    # B_BR is 1x1 (row 4, col 4)
    # C_R is 4x1
    # C[i, 4] += A[i, 4] B[4, 4]
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
