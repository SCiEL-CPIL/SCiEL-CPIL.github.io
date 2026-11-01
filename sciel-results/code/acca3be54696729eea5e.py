import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Coefficients for A (u) and B (v) for each term t
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    u_coeffs = [
        [1, 0, 0, 1], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [0, 0, 1, -1],# m6: c-a
        [0, 1, 0, -1] # m7: b-d
    ]
    
    v_coeffs = [
        [1, 0, 0, 1], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1],# m3: f-h
        [0, 0, 1, -1],# m4: g-e
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    
    # Coefficients for C (w) for each term t
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    w_coeffs = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1],# m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_blocks = p // 2
    for b in range(num_blocks):
        # Global column indices for B and C
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    U[idx_u][r] = u_coeffs[t][idx_u]
            
            # Fill V
            # V index: j * p + k
            for j in range(m):
                for k_local in range(2):
                    k_global = k0 + k_local
                    idx_v = j * p + k_global
                    # v_coeffs index: j * 2 + k_local
                    idx_vc = j * 2 + k_local
                    V[idx_v][r] = v_coeffs[t][idx_vc]
            
            # Fill W
            # W index: k * n + i
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(n):
                    idx_w = k_global * n + i
                    # w_coeffs index: i * 2 + k_local
                    idx_wc = i * 2 + k_local
                    W[idx_w][r] = w_coeffs[t][idx_wc]
                    
    return {"U": U, "V": V, "W": W}
