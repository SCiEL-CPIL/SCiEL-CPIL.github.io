DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U coeffs for A00, A01, A10, A11
    # V coeffs for B00, B01, B10, B11
    # W coeffs for C00, C01, C10, C11
    
    # Term 1: (A00 + A11)(B00 + B11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00, C01, C10, C11
    
    # Term 2: (A10 + A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # Term 3: A00(B01 - B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # Term 4: A11(B10 - B00)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0]
    
    # Term 5: (A00 + A01)B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # Term 6: (A10 - A00)(B00 + B01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # Term 7: (A01 - A11)(B10 + B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Map A indices to U rows: i*m + j
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # Map B indices to V rows: j*p + k
    # B00 (j=0, k=0) -> 0
    # B01 (j=0, k=1) -> 1
    # B10 (j=1, k=0) -> p
    # B11 (j=1, k=1) -> p+1
    
    # Map C indices to W rows: k*n + i
    # C00 (k=0, i=0) -> 0
    # C01 (k=1, i=0) -> 2
    # C10 (k=0, i=1) -> 1
    # C11 (k=1, i=1) -> 3
    
    # We have 8 blocks, each of size 2x2 in the k dimension.
    # Block b uses k in {2b, 2b+1}.
    
    for b in range(8):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for this block
        # B00: j=0, k=k0 -> 0*p + k0 = k0
        # B01: j=0, k=k1 -> k1
        # B10: j=1, k=k0 -> p + k0
        # B11: j=1, k=k1 -> p + k1
        
        v_idx = [k0, k1, p + k0, p + k1]
        
        # W indices for this block
        # C00: k=k0, i=0 -> k0*n + 0 = 2*k0
        # C01: k=k1, i=0 -> k1*n + 0 = 2*k1
        # C10: k=k0, i=1 -> k0*n + 1 = 2*k0 + 1
        # C11: k=k1, i=1 -> k1*n + 1 = 2*k1 + 1
        
        w_idx = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]
        
        # U indices are always 0, 1, 2, 3 for A00, A01, A10, A11
        u_idx = [0, 1, 2, 3]
        
        for t in range(7):
            col = b * 7 + t
            u, v, w = terms[t]
            
            for i in range(4):
                U[u_idx[i]][col] = u[i]
                V[v_idx[i]][col] = v[i]
                W[w_idx[i]][col] = w[i]
                
    return {"U": U, "V": V, "W": W}
