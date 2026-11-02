import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen bilinear forms for 2x2 matrices
    # w_terms: list of (lj, li, coeff)
    # lj is col index of C (0,1)
    # li is row index of C (0,1)
    # W index is lj * n + li
    
    forms = []
    
    # M1 = (x00 + x11)(y00 + y11)
    # C11 += 1, C22 += 1
    # C11: li=0, lj=0 -> (0,0)
    # C22: li=1, lj=1 -> (1,1)
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,1,1)]
    forms.append((u1, v1, w1))
    
    # M2 = (x10 + x11) y01
    # C12 += 1
    # C12: li=0, lj=1 -> (1,0)
    u2 = [(1,0,1), (1,1,1)]
    v2 = [(0,1,1)]
    w2 = [(1,0,1)]
    forms.append((u2, v2, w2))
    
    # M3 = x00 (y10 - y11)
    # C12 += 1, C22 -= 1
    # C12: li=0, lj=1 -> (1,0)
    # C22: li=1, lj=1 -> (1,1)
    u3 = [(0,0,1)]
    v3 = [(1,0,1), (1,1,-1)]
    w3 = [(1,0,1), (1,1,-1)]
    forms.append((u3, v3, w3))
    
    # M4 = (x00 - x11)(y00 + y01)
    # C11 += 1, C22 -= 1
    # C11: li=0, lj=0 -> (0,0)
    # C22: li=1, lj=1 -> (1,1)
    u4 = [(0,0,1), (1,1,-1)]
    v4 = [(0,0,1), (0,1,1)]
    w4 = [(0,0,1), (1,1,-1)]
    forms.append((u4, v4, w4))
    
    # M5 = (x01 + x11)(y10 + y11)
    # C11 -= 1, C21 += 1
    # C11: li=0, lj=0 -> (0,0)
    # C21: li=1, lj=0 -> (0,1)
    u5 = [(0,1,1), (1,1,1)]
    v5 = [(1,0,1), (1,1,1)]
    w5 = [(0,0,-1), (0,1,1)]
    forms.append((u5, v5, w5))
    
    # M6 = (x00 + x01)(y00 - y10)
    # C21 += 1, C22 += 1
    # C21: li=1, lj=0 -> (0,1)
    # C22: li=1, lj=1 -> (1,1)
    u6 = [(0,0,1), (0,1,1)]
    v6 = [(0,0,1), (1,0,-1)]
    w6 = [(0,1,1), (1,1,1)]
    forms.append((u6, v6, w6))
    
    # M7 = (x10 + x01)(y01 - y11)
    # C11 += 1
    # C11: li=0, lj=0 -> (0,0)
    u7 = [(1,0,1), (0,1,1)]
    v7 = [(0,1,1), (1,1,-1)]
    w7 = [(0,0,1)]
    forms.append((u7, v7, w7))

    def add_block(u_off, v_off, w_off, base_index):
        for f_idx in range(7):
            u_terms, v_terms, w_terms = forms[f_idx]
            idx = base_index + f_idx
            
            for (li, lk, cu) in u_terms:
                gi = u_off[0] + li
                gk = u_off[1] + lk
                U[gi * m + gk][idx] += cu
            
            for (lk, lj, cv) in v_terms:
                gk = v_off[0] + lk
                gj = v_off[1] + lj
                V[gk * p + gj][idx] += cv
            
            for (lj, li, cw) in w_terms:
                gj = w_off[0] + lj
                gi = w_off[1] + li
                W[gj * n + gi][idx] += cw

    # 1. A_L * B_LL -> C_L
    add_block((0, 0), (0, 0), (0, 0), 0)
    
    # 2. A_R * B_RL -> C_L
    add_block((0, 2), (2, 0), (0, 0), 7)
    
    # 3. A_L * B_LR -> C_R
    add_block((0, 0), (0, 2), (2, 0), 14)
    
    # 4. A_R * B_RR -> C_R
    add_block((0, 2), (2, 2), (2, 0), 21)
    
    return {"U": U, "V": V, "W": W}
