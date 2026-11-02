import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W as 2D lists of zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base 2x2 Strassen coefficients
    # U_base[k][i][j]
    U_base = [
        [[1, 0], [0, 1]], # 0
        [[0, 0], [1, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 0], [0, 1]], # 3
        [[1, 1], [0, 0]], # 4
        [[-1, 0], [1, 0]],# 5
        [[0, 1], [-1, 0]] # 6
    ]
    # V_base[k][j][k]
    V_base = [
        [[1, 0], [0, 1]], # 0
        [[1, 0], [0, 0]], # 1
        [[0, 1], [-1, 0]],# 2
        [[-1, 0], [1, 0]],# 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[0, 1], [0, 1]]  # 6
    ]
    # W_base[k][j][i] -> C[j][i]
    W_base = [
        [[1, 0], [0, 1]], # 0: C00, C11
        [[0, 0], [1, -1]],# 1: C10, C11(-)
        [[1, 0], [1, 0]], # 2: C01, C11
        [[1, 0], [1, 0]], # 3: C00, C10
        [[-1, 1], [0, 0]],# 4: C00(-), C01
        [[0, 0], [0, 1]], # 5: C11
        [[1, 0], [-1, 0]] # 6: C00, C11(-)
    ]
    
    # Outer 4x4 Strassen coefficients (Block level)
    # U_outer[K][a][b] where A_ab is block
    U_outer = [
        [[1, 0], [0, 1]], # 0: A00, A11
        [[0, 0], [1, 1]], # 1: A10, A11
        [[1, 0], [0, 0]], # 2: A00
        [[0, 0], [0, 1]], # 3: A11
        [[1, 1], [0, 0]], # 4: A00, A01
        [[-1, 0], [1, 0]],# 5: A00(-), A10
        [[0, 1], [-1, 0]] # 6: A01, A11(-)
    ]
    # V_outer[K][c][d] where B_cd is block
    V_outer = [
        [[1, 0], [0, 1]], # 0: B00, B11
        [[1, 0], [0, 0]], # 1: B00
        [[0, 1], [-1, 0]],# 2: B01, B11(-)
        [[-1, 0], [1, 0]],# 3: B00(-), B10
        [[0, 0], [0, 1]], # 4: B11
        [[1, 1], [0, 0]], # 5: B00, B01
        [[0, 1], [0, 1]]  # 6: B10, B11
    ]
    # W_outer[K][A][B] where C_AB is block
    W_outer = [
        [[1, 0], [0, 1]], # 0: C00, C11
        [[0, 0], [1, -1]],# 1: C10, C11(-)
        [[1, 0], [1, 0]], # 2: C01, C11
        [[1, 0], [1, 0]], # 3: C00, C10
        [[-1, 1], [0, 0]],# 4: C00(-), C01
        [[0, 0], [0, 1]], # 5: C11
        [[1, 0], [-1, 0]] # 6: C00, C11(-)
    ]

    # Generate rank 49 decomposition
    for K in range(7):
        for k in range(7):
            r = 7 * K + k
            
            # Update U
            for i in range(2):
                for j in range(2):
                    u_inner = U_base[k][i][j]
                    if u_inner == 0: continue
                    for a in range(2):
                        for b in range(2):
                            u_outer = U_outer[K][a][b]
                            if u_outer == 0: continue
                            global_i = 2 * a + i
                            global_j = 2 * b + j
                            U[global_i * m + global_j][r] += u_inner * u_outer
                            
            # Update V
            for j in range(2):
                for k_idx in range(2):
                    v_inner = V_base[k][j][k_idx]
                    if v_inner == 0: continue
                    for c in range(2):
                        for d in range(2):
                            v_outer = V_outer[K][c][d]
                            if v_outer == 0: continue
                            global_j = 2 * c + j
                            global_k = 2 * d + k_idx
                            V[global_j * p + global_k][r] += v_inner * v_outer
                            
            # Update W
            for i in range(2):
                for j in range(2):
                    w_inner = W_base[k][j][i]
                    if w_inner == 0: continue
                    for A in range(2):
                        for B in range(2):
                            w_outer = W_outer[K][A][B]
                            if w_outer == 0: continue
                            global_i = 2 * A + i
                            global_j = 2 * B + j
                            # W index: k*n + i -> global_j * n + global_i
                            W[global_j * n + global_i][r] += w_inner * w_outer

    return {"U": U, "V": V, "W": W}
