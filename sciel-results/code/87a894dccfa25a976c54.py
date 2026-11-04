import numpy as np

def construct():
    # Base Strassen vectors for 2x2x2
    # u: A indices (i*2+k), v: B indices (k*2+j), w: C indices (j*2+i)
    
    # P1 = (A00+A11)(B00+B11)
    u1 = np.array([1, 0, 0, 1], dtype=float)
    v1 = np.array([1, 0, 0, 1], dtype=float)
    w1 = np.array([1, 0, 0, 1], dtype=float) # C00:+1, C11:+1
    
    # P2 = (A10+A11)B00
    u2 = np.array([0, 1, 0, 1], dtype=float)
    v2 = np.array([1, 0, 0, 0], dtype=float)
    w2 = np.array([0, 0, 1, -1], dtype=float) # C10:+1, C11:-1
    
    # P3 = A00(B01-B11)
    u3 = np.array([1, 0, 0, 0], dtype=float)
    v3 = np.array([0, 1, 0, -1], dtype=float)
    w3 = np.array([0, 1, 0, 1], dtype=float) # C01:+1, C11:+1
    
    # P4 = A11(B10-B00)
    u4 = np.array([0, 0, 0, 1], dtype=float)
    v4 = np.array([-1, 0, 1, 0], dtype=float) # -B00+B10
    w4 = np.array([1, 0, 1, 0], dtype=float) # C00:+1, C10:+1
    
    # P5 = (A00+A10)B11
    u5 = np.array([1, 1, 0, 0], dtype=float)
    v5 = np.array([0, 0, 0, 1], dtype=float)
    w5 = np.array([-1, 1, 0, 0], dtype=float) # C00:-1, C01:+1
    
    # P6 = (A10-A00)(B00+B01)
    u6 = np.array([-1, 1, 0, 0], dtype=float)
    v6 = np.array([1, 1, 0, 0], dtype=float)
    w6 = np.array([0, 0, 0, 1], dtype=float) # C11:+1
    
    # P7 = (A01-A11)(B10+B11)
    u7 = np.array([0, 1, 0, -1], dtype=float)
    v7 = np.array([0, 0, 1, 1], dtype=float)
    w7 = np.array([-1, 0, 0, 0], dtype=float) # C00:-1
    
    base_u = [u1, u2, u3, u4, u5, u6, u7]
    base_v = [v1, v2, v3, v4, v5, v6, v7]
    base_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Generate 49 terms for 4x4x4
    # u4x4: 16 elements (i*4+k)
    # v4x4: 16 elements (k*4+j)
    # w4x4: 16 elements (j*4+i)
    
    terms_4x4 = []
    for s1 in range(7):
        for s2 in range(7):
            u_4x4 = np.kron(base_u[s1], base_u[s2])
            v_4x4 = np.kron(base_v[s1], base_v[s2])
            w_4x4 = np.kron(base_w[s1], base_w[s2])
            terms_4x4.append((u_4x4, v_4x4, w_4x4))
            
    # Construct 98 terms for 4x4x8
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for idx, (u_4x4, v_4x4, w_4x4) in enumerate(terms_4x4):
        # First block (j=0..3)
        # U is same
        U[:, idx] = u_4x4
        
        # V: index k*8+j. v_4x4 index k*4+j.
        # For k in 0..3, j in 0..3:
        # V[k*8 + j] = v_4x4[k*4 + j]
        for k in range(4):
            for j in range(4):
                V[k * 8 + j, idx] = v_4x4[k * 4 + j]
        # V[k*8 + 4 + j] = 0 (already 0)
        
        # W: index j*4+i. w_4x4 index j*4+i.
        # For j in 0..3, i in 0..3:
        # W[j*4 + i] = w_4x4[j*4 + i]
        W[:16, idx] = w_4x4
        # W[16:32, idx] = 0 (already 0)
        
        # Second block (j=4..7)
        idx2 = idx + 49
        U[:, idx2] = u_4x4
        
        # V: index k*8+j. 
        # For k in 0..3, j in 0..3:
        # V[k*8 + 4 + j] = v_4x4[k*4 + j]
        for k in range(4):
            for j in range(4):
                V[k * 8 + 4 + j, idx2] = v_4x4[k * 4 + j]
        # V[k*8 + j] = 0 (already 0)
        
        # W: index j*4+i.
        # For j in 0..3, i in 0..3:
        # W[(4+j)*4 + i] = w_4x4[j*4 + i]
        # This maps to indices 16..31
        W[16:32, idx2] = w_4x4
        # W[:16, idx2] = 0 (already 0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
