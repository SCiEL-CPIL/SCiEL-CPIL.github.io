import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85

    # 2x2 Strassen factors (7 terms)
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    col = 0

    # 4x4x4 part: 49 terms (7 outer x 7 inner)
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        for inner_k in range(7):
            col_idx = col
            col += 1
            u_vec = s_u[inner_k]
            v_vec = s_v[inner_k]
            w_vec = s_w[inner_k]
            
            for (a_br, a_bc), a_c in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        val = u_vec[li]
                        if val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj][col_idx] += a_c * val
            
            for (b_br, b_bc), b_c in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        val = v_vec[li]
                        if val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk][col_idx] += b_c * val
            
            for (c_br, c_bc), c_c in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        val = w_vec[li]
                        if val != 0:
                            gk = 2 * c_bc + lc
                            gi = 2 * c_br + lr
                            W[gk * 4 + gi][col_idx] += c_c * val

    # 4x4x1 part: 16 terms
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # 4x1x4 part: 16 terms
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1

    # 4x1x1 part: 4 terms
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
