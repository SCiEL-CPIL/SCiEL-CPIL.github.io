import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # A: 2x2, B: 2x12, C: 2x12 (stored transposed)
    # U: (4, rank), V: (24, rank), W: (24, rank)
    
    num_blocks = p // 2  # 6 blocks of 2x2
    rank_per_block = 7  # Strassen
    rank = num_blocks * rank_per_block  # 42
    
    U = np.zeros((n * m, rank))  # (4, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (24, 42)
    
    # Strassen multiplications for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M3
    
    # u vectors (4-dim: [a, b, c, d])
    u_vectors = {
        1: [1, 0, 0, 1],   # a+d
        2: [0, 0, 1, 1],   # c+d
        3: [1, 0, 0, 0],   # a
        4: [0, 0, 0, 1],   # d
        5: [1, 1, 0, 0],   # a+b
        6: [-1, 0, 1, 0],  # c-a
        7: [0, 1, 0, -1],  # b-d
    }
    
    # For each block b, define v and w vectors for each Strassen term
    for b in range(num_blocks):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        # B indices: e=B[0,col0], f=B[0,col1], g=B[1,col0], h=B[1,col1]
        # B flattened: pos = j*p + k
        e_idx = 0 * p + col0
        f_idx = 0 * p + col1
        g_idx = 1 * p + col0
        h_idx = 1 * p + col1
        
        # C indices (transposed storage: W[k*n+i]):
        # C[0,col0] -> W[col0*n + 0]
        # C[0,col1] -> W[col1*n + 0]
        # C[1,col0] -> W[col0*n + 1]
        # C[1,col1] -> W[col1*n + 1]
        c00_idx = col0 * n + 0
        c01_idx = col1 * n + 0
        c10_idx = col0 * n + 1
        c11_idx = col1 * n + 1
        
        # v vectors (24-dim, non-zero only in block b)
        # w vectors (24-dim, non-zero only in block b)
        
        # M1: v = e+h, w = C11 + C22
        v1 = np.zeros(m * p)
        v1[e_idx] = 1
        v1[h_idx] = 1
        w1 = np.zeros(p * n)
        w1[c00_idx] = 1
        w1[c11_idx] = 1
        
        # M2: v = e, w = C21
        v2 = np.zeros(m * p)
        v2[e_idx] = 1
        w2 = np.zeros(p * n)
        w2[c10_idx] = 1
        
        # M3: v = f-h, w = C12 + C22
        v3 = np.zeros(m * p)
        v3[f_idx] = 1
        v3[h_idx] = -1
        w3 = np.zeros(p * n)
        w3[c01_idx] = 1
        w3[c11_idx] = 1
        
        # M4: v = g-e, w = C11 + C21
        v4 = np.zeros(m * p)
        v4[g_idx] = 1
        v4[e_idx] = -1
        w4 = np.zeros(p * n)
        w4[c00_idx] = 1
        w4[c10_idx] = 1
        
        # M5: v = h, w = -C11 + C12
        v5 = np.zeros(m * p)
        v5[h_idx] = 1
        w5 = np.zeros(p * n)
        w5[c00_idx] = -1
        w5[c01_idx] = 1
        
        # M6: v = e+f, w = C22
        v6 = np.zeros(m * p)
        v6[e_idx] = 1
        v6[f_idx] = 1
        w6 = np.zeros(p * n)
        w6[c11_idx] = 1
        
        # M7: v = g+h, w = C11
        v7 = np.zeros(m * p)
        v7[g_idx] = 1
        v7[h_idx] = 1
        w7 = np.zeros(p * n)
        w7[c00_idx] = 1
        
        v_list = [v1, v2, v3, v4, v5, v6, v7]
        w_list = [w1, w2, w3, w4, w5, w6, w7]
        
        for t in range(7):
            col = b * 7 + t
            U[:, col] = u_vectors[t + 1]
            V[:, col] = v_list[t]
            W[:, col] = w_list[t]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
