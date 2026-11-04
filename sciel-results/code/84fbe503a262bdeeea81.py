import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2x2 Strassen factors (u, v, w are 4-vectors)
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    # Top-level 4x4x4 Strassen block coefficients
    # U_top[r] is a 2x2 matrix of coefficients for the 2x2 blocks of A
    U_top = [
        [[1, 0], [0, 1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]],
        [[1, 1], [0, 0]], [[-1, 0], [1, 0]], [[0, 1], [0, -1]]
    ]
    V_top = [
        [[1, 0], [0, 1]], [[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, -1], [1, 0]],
        [[0, 0], [0, 1]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]
    ]
    W_top = [
        [[1, 0], [0, 1]], [[0, 0], [1, -1]], [[0, 1], [0, 1]], [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]]
    ]
    
    t = 0
    for r in range(7):
        for s in range(7):
            # Construct U vector (length 16)
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # Block 00
            c_u = U_top[r][0][0]
            c_v = V_top[r][0][0]
            c_w = W_top[r][0][0]
            if c_u != 0:
                for i in range(4): u_vec[i] += c_u * S_U[s][i]
            if c_v != 0:
                for i in range(4): v_vec[i] += c_v * S_V[s][i]
            if c_w != 0:
                for i in range(4): w_vec[i] += c_w * S_W[s][i]
                
            # Block 01
            c_u = U_top[r][0][1]
            c_v = V_top[r][0][1]
            c_w = W_top[r][0][1]
            if c_u != 0:
                for i in range(4): u_vec[4+i] += c_u * S_U[s][i]
            if c_v != 0:
                for i in range(4): v_vec[4+i] += c_v * S_V[s][i]
            if c_w != 0:
                for i in range(4): w_vec[4+i] += c_w * S_W[s][i]
                
            # Block 10
            c_u = U_top[r][1][0]
            c_v = V_top[r][1][0]
            c_w = W_top[r][1][0]
            if c_u != 0:
                for i in range(4): u_vec[8+i] += c_u * S_U[s][i]
            if c_v != 0:
                for i in range(4): v_vec[8+i] += c_v * S_V[s][i]
            if c_w != 0:
                for i in range(4): w_vec[8+i] += c_w * S_W[s][i]
                
            # Block 11
            c_u = U_top[r][1][1]
            c_v = V_top[r][1][1]
            c_w = W_top[r][1][1]
            if c_u != 0:
                for i in range(4): u_vec[12+i] += c_u * S_U[s][i]
            if c_v != 0:
                for i in range(4): v_vec[12+i] += c_v * S_V[s][i]
            if c_w != 0:
                for i in range(4): w_vec[12+i] += c_w * S_W[s][i]
                
            # Fill U, V, W
            for i in range(16):
                U[i][t] = u_vec[i]
                V[i][t] = v_vec[i]
                W[i][t] = w_vec[i]
            t += 1
            
    # Part 2: 4x4x3 Trivial (cols 4-6 of B)
    # i in 0..3, j in 0..3, k in 4..6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
