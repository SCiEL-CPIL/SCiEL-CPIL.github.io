import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for the rank-49 decomposition (2-level Strassen)
    # Outer Strassen (block level)
    U_A = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: 1},
        3: {0: 1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 0: -1},
        7: {1: 1, 3: -1}
    }
    V_B = {
        1: {0: 1, 3: 1},
        2: {0: 1},
        3: {1: 1, 3: -1},
        4: {2: 1, 0: -1},
        5: {3: 1},
        6: {0: 1, 1: 1},
        7: {2: 1, 3: 1}
    }
    
    # Inner Strassen (scalar level)
    U_x = {
        1: [1, 0, 0, 1],
        2: [0, 0, 1, 1],
        3: [1, 0, 0, 0],
        4: [0, 0, 0, 1],
        5: [1, 1, 0, 0],
        6: [-1, 0, 1, 0],
        7: [0, 1, 0, -1]
    }
    V_y = {
        1: [1, 0, 0, 1],
        2: [1, 0, 0, 0],
        3: [0, 1, 0, -1],
        4: [-1, 0, 1, 0],
        5: [0, 0, 0, 1],
        6: [1, 1, 0, 0],
        7: [0, 0, 1, 1]
    }
    
    W_inner = {
        (1, 0, 0): 1, (4, 0, 0): 1, (5, 0, 0): -1, (7, 0, 0): 1,
        (3, 0, 1): 1, (5, 0, 1): 1,
        (2, 1, 0): 1, (4, 1, 0): 1,
        (1, 1, 1): 1, (2, 1, 1): -1, (3, 1, 1): 1, (6, 1, 1): 1
    }
    
    W_outer = {
        0: {1: 1, 4: 1, 5: -1, 7: 1},
        1: {3: 1, 5: 1},
        2: {2: 1, 4: 1},
        3: {1: 1, 2: -1, 3: 1, 6: 1}
    }
    
    for alpha in range(1, 8):
        for beta in range(1, 8):
            c = (alpha - 1) * 7 + (beta - 1)
            
            # U
            for r in range(2):
                for cc in range(2):
                    ux_val = U_x[beta][r * 2 + cc]
                    if ux_val == 0: continue
                    for br in range(2):
                        for bc in range(2):
                            ua_val = U_A[alpha].get(br * 2 + bc, 0)
                            if ua_val == 0: continue
                            i = 2 * br + r
                            j = 2 * bc + cc
                            U[i * 4 + j][c] += ua_val * ux_val
                            
            # V
            for r in range(2):
                for cc in range(2):
                    vy_val = V_y[beta][r * 2 + cc]
                    if vy_val == 0: continue
                    for br in range(2):
                        for bc in range(2):
                            vb_val = V_B[alpha].get(br * 2 + bc, 0)
                            if vb_val == 0: continue
                            j = 2 * br + r
                            k = 2 * bc + cc
                            V[j * 4 + k][c] += vb_val * vy_val
                            
            # W
            for i in range(4):
                for k in range(4):
                    br_out = i // 2
                    r = i % 2
                    bc_out = k // 2
                    cc = k % 2
                    block_out = br_out * 2 + bc_out
                    wo_val = W_outer[block_out].get(alpha, 0)
                    wi_val = W_inner.get((beta, r, cc), 0)
                    if wo_val != 0 and wi_val != 0:
                        W[k * 4 + i][c] += wo_val * wi_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
