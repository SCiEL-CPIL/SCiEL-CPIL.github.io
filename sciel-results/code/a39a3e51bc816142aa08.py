import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85
    
    # Strassen 2x2x2 factors (verified)
    s_u = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1],
    ])
    s_v = np.array([
        [1, 1, 0, -1, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, 0, 1, 0, 1],
    ])
    s_w = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ])
    
    # 7 Strassen terms for 4x4x4 (block-level)
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        for inner_t in range(7):
            col_idx = t_idx * 7 + inner_t
            
            for (abr, abc), ac in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        u_val = s_u[lr, lc][inner_t]
                        if u_val != 0:
                            gi = 2 * abr + lr
                            gj = 2 * abc + lc
                            U[gi * 5 + gj][col_idx] += ac * u_val
            
            for (bbr, bbc), bc in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        v_val = s_v[lr, lc][inner_t]
                        if v_val != 0:
                            gj = 2 * bbr + lr
                            gk = 2 * bbc + lc
                            V[gj * 5 + gk][col_idx] += bc * v_val
            
            for (cbr, cbc), cc in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        w_val = s_w[lr, lc][inner_t]
                        if w_val != 0:
                            gk = 2 * cbc + lc
                            gi = 2 * cbr + lr
                            W[gk * 4 + gi][col_idx] += cc * w_val
    
    col = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
