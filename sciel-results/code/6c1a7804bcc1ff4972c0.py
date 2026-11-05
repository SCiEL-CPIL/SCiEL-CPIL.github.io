import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # U_S, V_S, W_S are 4x7
    U_S = [
        [0, 0, 1, 0, 1, 1, 0], # a11
        [0, 0, 0, 0, 1, 0, 1], # a12
        [0, 1, 0, 0, 0, 1, 0], # a21
        [0, 1, 0, 1, 0, 0, 1]  # a22
    ]
    V_S = [
        [1, 1, 0, 1, 0, 1, 0], # b11
        [0, 0, 1, 0, 0, 1, 0], # b12
        [0, 0, 0, 1, 1, 0, 1], # b21
        [1, 0, 0, 0, 1, 0, 1]  # b22
    ]
    W_S = [
        [1, 0, 0, 1, 0, 1, 0], # c11
        [0, 0, 1, 0, 1, 0, 0], # c12
        [0, 1, 0, 1, 0, 0, 0], # c21
        [1, 0, 1, 0, 0, 0, 1]  # c22
    ]
    
    # R factors for 1x2 * 2x2
    # U_R is 2x4, V_R is 4x4, W_R is 2x4
    # Terms:
    # 1: a1 b11 c1
    # 2: a2 b21 c1
    # 3: a1 b12 c2
    # 4: a2 b22 c2
    U_R = [
        [1, 0, 1, 0], # a1
        [0, 1, 0, 1]  # a2
    ]
    V_R = [
        [1, 0, 0, 0], # b11
        [0, 1, 0, 0], # b12
        [0, 0, 1, 0], # b21
        [0, 0, 0, 1]  # b22
    ]
    W_R = [
        [1, 1, 0, 0], # c1
        [0, 0, 1, 1]  # c2
    ]
    
    col = 0
    for b in range(2):
        for s in range(7):
            for r in range(4):
                # Get indices from Strassen
                u_s = 0
                for i in range(4):
                    if U_S[i][s] == 1:
                        u_s = i
                        break
                v_s = 0
                for i in range(4):
                    if V_S[i][s] == 1:
                        v_s = i
                        break
                w_s = 0
                for i in range(4):
                    if W_S[i][s] == 1:
                        w_s = i
                        break
                        
                # Get indices from R
                u_r = 0
                for i in range(2):
                    if U_R[i][r] == 1:
                        u_r = i
                        break
                v_r = 0
                for i in range(4):
                    if V_R[i][r] == 1:
                        v_r = i
                        break
                w_r = 0
                for i in range(2):
                    if W_R[i][r] == 1:
                        w_r = i
                        break
                        
                # Calculate global indices
                # U
                i_s = u_s // 2
                j_s = u_s % 2
                i = i_s
                j = 2 * j_s + u_r
                idx_u = i * 4 + j
                
                # V
                i_vs = v_s // 2
                j_vs = v_s % 2
                i_vr = v_r // 2
                j_vr = v_r % 2
                i = 2 * i_vs + i_vr
                j = 2 * (j_vs + 2 * b) + j_vr
                idx_v = i * 8 + j
                
                # W
                i_ws = w_s // 2
                j_ws = w_s % 2
                i = i_ws
                j = 2 * (j_ws + 2 * b) + w_r
                idx_w = i * 8 + j
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
