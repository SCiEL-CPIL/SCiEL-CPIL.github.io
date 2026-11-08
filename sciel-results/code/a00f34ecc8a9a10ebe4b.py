import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Hybrid: Strassen 4x4x4 (rank 49) + Schoolbook 4x4x3 (rank 48) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 inner coefficients
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Outer Strassen 4x4x4 coefficients (same structure as inner)
    u_out = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_out = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 28
            w_vec = [0] * 28
            
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k][b]
                        if co == 0:
                            continue
                        br = b // 2
                        bc = b % 2
                        ig = 2 * br + i_inner
                        jg = 2 * bc + j_inner
                        u_vec[ig * 4 + jg] += ci * co
            
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k][b]
                        if co == 0:
                            continue
                        br = b // 2
                        bc = b % 2
                        jg = 2 * br + i_inner
                        kg = 2 * bc + j_inner
                        v_vec[jg * 7 + kg] += ci * co
            
            for cb in range(4):
                cow = w_outer[cb][k]
                if cow == 0:
                    continue
                for ci2 in range(4):
                    ciw = w_out[ci2][l]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    cb_r = cb // 2
                    cb_c = cb % 2
                    ci_r = ci2 // 2
                    ci_c = ci2 % 2
                    ig = 2 * cb_r + ci_r
                    kg = 2 * cb_c + ci_c
                    w_vec[kg * 4 + ig] += tc
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
                W[r][col] = w_vec[r]
            col += 1
    
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                kg = 4 + kp
                U[i * 4 + j][col] = 1
                V[j * 7 + kg][col] = 1
                W[kg * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
