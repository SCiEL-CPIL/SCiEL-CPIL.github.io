import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients
    # alpha in 1..7
    # U_A[alpha, block] where block in 00, 01, 10, 11
    # block index 0:00, 1:01, 2:10, 3:11
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
    
    # Inner Strassen coefficients
    # beta in 1..7
    # U_x[beta, i] where i in 0..3 (00, 01, 10, 11)
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
    
    # Inner W coefficients: W_inner[beta, r, c]
    W_inner = {}
    W_inner[(1, 0, 0)] = 1
    W_inner[(4, 0, 0)] = 1
    W_inner[(5, 0, 0)] = -1
    W_inner[(7, 0, 0)] = 1
    
    W_inner[(3, 0, 1)] = 1
    W_inner[(5, 0, 1)] = 1
    
    W_inner[(2, 1, 0)] = 1
    W_inner[(4, 1, 0)] = 1
    
    W_inner[(1, 1, 1)] = 1
    W_inner[(2, 1, 1)] = -1
    W_inner[(3, 1, 1)] = 1
    W_inner[(6, 1, 1)] = 1
    
    # Outer W coefficients: W_outer[block, alpha]
    # block in 0..3 (00, 01, 10, 11)
    W_outer = {
        0: {1: 1, 4: 1, 5: -1, 7: 1}, # 00
        1: {3: 1, 5: 1},               # 01
        2: {2: 1, 4: 1},               # 10
        3: {1: 1, 2: -1, 3: 1, 6: 1}   # 11
    }
    
    # Construct U, V, W
    for alpha in range(1, 8):
        for beta in range(1, 8):
            c = (alpha - 1) * 7 + (beta - 1)
            
            # U[i*4+j, c]
            # U[i*4+j, c] = sum_{br, bc} U_A[alpha, br*2+bc] * U_x[beta, r*2+c]
            # where i = 2*br + r, j = 2*bc + cc
            # Note: U_x index is r*2+cc
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
                            
            # V[j*4+k, c]
            # V[j*4+k, c] = sum_{br, bc} V_B[alpha, br*2+bc] * V_y[beta, r*2+c]
            # where j = 2*br + r, k = 2*bc + cc
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
                            
            # W[k*4+i, c]
            # W[k*4+i, c] = W_outer[block_out, alpha] * W_inner[beta, r, cc]
            # block_out is determined by i, k
            # i = 2*br_out + r, k = 2*bc_out + cc
            # block_out = br_out * 2 + bc_out
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
