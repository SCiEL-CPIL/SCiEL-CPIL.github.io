import numpy as np

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen setup
    # U vectors for a,b,c,d (indices 0,1,2,3)
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [-1, 0, 1, 0]
    u7 = [0, 1, 0, -1]
    
    # V vectors for e,f,g,h (local indices 0,1,2,3)
    v1 = [1, 0, 1, 0]
    v2 = [1, 0, 0, 0]
    v3 = [0, 1, -1, 0]
    v4 = [0, 0, 1, -1]
    v5 = [0, 0, 0, 1]
    v6 = [1, 1, 0, 0]
    v7 = [0, 1, 0, 1]
    
    us = [u1, u2, u3, u4, u5, u6, u7]
    vs = [v1, v2, v3, v4, v5, v6, v7]
    
    # Compute W coefficients
    # Target matrices for C00, C01, C10, C11 in terms of a,b,c,d and e,f,g,h
    # C00 = ae + bg -> 1 at (0,0) and (0,2)
    # C01 = af + bh -> 1 at (0,1) and (0,3)
    # C10 = ce + dg -> 1 at (1,0) and (1,2)
    # C11 = cf + dh -> 1 at (1,1) and (1,3)
    
    targets = [
        np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).flatten(), # C00
        np.array([[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).flatten(), # C01
        np.array([[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).flatten(), # C10
        np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]).flatten()  # C11
    ]
    
    T_flat = np.zeros((16, 7))
    for i in range(7):
        T_flat[:, i] = np.outer(us[i], vs[i]).flatten()
        
    w_coeffs = []
    for t in targets:
        w = np.linalg.lstsq(T_flat, t, rcond=None)[0]
        # Round to nearest integer
        w = np.round(w)
        w_coeffs.append(w)
        
    # w_coeffs[0] is for C00, w_coeffs[1] for C01, etc.
    # Each w is a vector of 7 coefficients for M1..M7.
    
    col = 0
    for g in range(7):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        for i in range(7):
            # U part
            for u_idx in range(4):
                U[u_idx][col] = us[i][u_idx]
                
            # V part
            # e is B[0, k0], f is B[0, k1], g is B[1, k0], h is B[1, k1]
            v_idx_map = {
                0: 0 * p + k0, # e
                1: 0 * p + k1, # f
                2: 1 * p + k0, # g
                3: 1 * p + k1  # h
            }
            for v_local_idx in range(4):
                v_global_idx = v_idx_map[v_local_idx]
                V[v_global_idx][col] = vs[i][v_local_idx]
                
            # W part
            # C00 is W[k0*n + 0], C10 is W[k0*n + 1]
            # C01 is W[k1*n + 0], C11 is W[k1*n + 1]
            # w_coeffs[0] is for C00, w_coeffs[1] for C01, w_coeffs[2] for C10, w_coeffs[3] for C11
            # Wait, my targets were:
            # 0: C00
            # 1: C01
            # 2: C10
            # 3: C11
            
            W[k0 * n + 0][col] = w_coeffs[0][i] # C00
            W[k1 * n + 0][col] = w_coeffs[1][i] # C01
            W[k0 * n + 1][col] = w_coeffs[2][i] # C10
            W[k1 * n + 1][col] = w_coeffs[3][i] # C11
            
            col += 1
            
    # Last column (k=14)
    # Schoolbook 2x2 x 2x1
    # C0 = a*x + b*y
    # C1 = c*x + d*y
    # x = B[0, 14], y = B[1, 14]
    
    # Term 1: a*x
    U[0][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 0][col] = 1 # C0
    W[14 * n + 1][col] = 0
    col += 1
    
    # Term 2: b*y
    U[1][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 0][col] = 1 # C0
    W[14 * n + 1][col] = 0
    col += 1
    
    # Term 3: c*x
    U[2][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 0][col] = 0
    W[14 * n + 1][col] = 1 # C1
    col += 1
    
    # Term 4: d*y
    U[3][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 0][col] = 0
    W[14 * n + 1][col] = 1 # C1
    col += 1
    
    return {"U": U, "V": V, "W": W}
