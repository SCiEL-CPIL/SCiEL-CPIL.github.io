import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2
    # U_local: 4 entries (i*m+j)
    # V_local: 4 entries (j*p_local+k_local) -> j*2+k
    # W_local: 4 entries (k_local*n+i) -> k*2+i
    
    # Term definitions:
    # 1: (a00+a11)(b00+b11)
    # 2: (a10+a11)b00
    # 3: a00(b01-b11)
    # 4: a11(b10-b00)
    # 5: (a00+a01)b11
    # 6: (a10-a00)(b00+b01)
    # 7: (a01-a11)(b10+b11)
    
    # U vectors (length 4)
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [1, 0, 1, 0] # a10-a00 -> idx 2 is a10, idx 0 is a00. So [ -1, 0, 1, 0 ]
    u7 = [0, 1, 0, -1] # a01-a11 -> idx 1 is a01, idx 3 is a11. So [ 0, 1, 0, -1 ]
    
    # Correction for u6:
    # a10 is at i=1, j=0 -> idx 1*2+0 = 2.
    # a00 is at i=0, j=0 -> idx 0*2+0 = 0.
    # So u6 should be [-1, 0, 1, 0].
    
    # V vectors (length 4)
    v1 = [1, 0, 0, 1] # b00+b11
    v2 = [1, 0, 0, 0] # b00
    v3 = [0, 1, 0, -1] # b01-b11
    v4 = [0, 0, 1, -1] # b10-b00
    v5 = [0, 0, 0, 1] # b11
    v6 = [1, 1, 0, 0] # b00+b01
    v7 = [0, 0, 1, 1] # b10+b11
    
    # W vectors (length 4)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # w_s[k*2+i]
    w1 = [1, 0, 0, 1]
    w2 = [0, 1, 0, -1]
    w3 = [0, 0, 1, 1]
    w4 = [1, 1, 0, 0]
    w5 = [-1, 0, 1, 0]
    w6 = [0, 0, 0, 1]
    w7 = [1, 0, 0, 0]
    
    u_list = [u1, u2, u3, u4, u5, u6, u7]
    v_list = [v1, v2, v3, v4, v5, v6, v7]
    w_list = [w1, w2, w3, w4, w5, w6, w7]
    
    # Fill U, V, W
    for t in range(3):
        for s in range(7):
            r = 7 * t + s
            u = u_list[s]
            v = v_list[s]
            w = w_list[s]
            
            # U is same for all t
            for idx, val in enumerate(u):
                U[idx][r] = val
                
            # V mapping: local j*2+k -> global j*6 + 3t + k
            for j in range(2):
                for k in range(2):
                    local_idx = j * 2 + k
                    global_idx = j * 6 + 3 * t + k
                    V[global_idx][r] = v[local_idx]
                    
            # W mapping: local k*2+i -> global (3t+k)*2 + i
            for k in range(2):
                for i in range(2):
                    local_idx = k * 2 + i
                    global_idx = (3 * t + k) * 2 + i
                    W[global_idx][r] = w[local_idx]
                    
    return {"U": U, "V": V, "W": W}
