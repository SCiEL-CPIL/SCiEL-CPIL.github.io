import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2
    # u_s, v_s, w_s are 4-vectors (flattened 2x2)
    # Indexing: [00, 01, 10, 11] -> [0, 1, 2, 3]
    u_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    v_strassen = [
        [1, 1, 0, 0], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, -1, 1, 0], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    w_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # Standard decomposition for 3x3
    # a_t, b_t, c_t are 9-vectors (flattened 3x3)
    # Term t corresponds to (i, k) for A, (k, j) for B, (i, j) for C
    # i, j, k in 0..2
    a_std = []
    b_std = []
    c_std = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = [0] * 9
                b = [0] * 9
                c = [0] * 9
                a[i * 3 + k] = 1
                b[k * 3 + j] = 1
                c[i * 3 + j] = 1
                a_std.append(a)
                b_std.append(b)
                c_std.append(c)
                
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    for s in range(7):
        for t in range(27):
            # Extract indices from t
            # t = i*9 + j*3 + k
            i = t // 9
            rem = t % 9
            j = rem // 3
            k = rem % 3
            
            # a_t[i, k] = 1
            # b_t[k, j] = 1
            # c_t[i, j] = 1
            
            for alpha in range(2):
                for beta in range(2):
                    # U index: (2i+alpha, 2j+beta)
                    row = 2 * i + alpha
                    col_u = 2 * j + beta
                    val = 1 * u_strassen[s][alpha * 2 + beta]
                    if val != 0:
                        U[row * 6 + col_u][col] = val
                        
                for gamma in range(2):
                    # V index: (2j+beta, 2k+gamma)
                    # Wait, V depends on beta too.
                    # We need to loop beta as well for V.
                    pass
            
            # Let's restructure the loops for clarity
            # U depends on i, alpha, j, beta
            # V depends on j, beta, k, gamma
            # W depends on i, alpha, k, gamma
            
            # U
            for alpha in range(2):
                for beta in range(2):
                    row = 2 * i + alpha
                    col_u = 2 * j + beta
                    val = u_strassen[s][alpha * 2 + beta]
                    if val != 0:
                        U[row * 6 + col_u][col] = val
                        
            # V
            for beta in range(2):
                for gamma in range(2):
                    row = 2 * j + beta
                    col_v = 2 * k + gamma
                    val = v_strassen[s][beta * 2 + gamma]
                    if val != 0:
                        V[row * 6 + col_v][col] = val
                        
            # W
            for alpha in range(2):
                for gamma in range(2):
                    row = 2 * i + alpha
                    col_w = 2 * k + gamma
                    val = w_strassen[s][alpha * 2 + gamma]
                    if val != 0:
                        W[row * 6 + col_w][col] = val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
