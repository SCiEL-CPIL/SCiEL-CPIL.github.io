import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Rank 56 Strassen decomposition
    # Split m=4 into two blocks of size 2.
    # For each block, we have a 2x2 * 2x8 multiplication.
    # 2x2 * 2x8 is equivalent to 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 has rank 7 (Strassen).
    # Total rank = 2 (m-blocks) * 4 (p-blocks) * 7 (Strassen) = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen forms for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    X_forms = np.array([
        [1, 0, 0, 1],    # a+d
        [0, 0, 1, 1],    # c+d
        [1, 0, 0, 0],    # a
        [0, 0, 0, 1],    # d
        [1, 1, 0, 0],    # a+b
        [-1, 0, 1, 0],   # c-a
        [0, 1, 0, -1],   # b-d
    ], dtype=np.int64)
    
    Y_forms = np.array([
        [1, 0, 0, 1],    # e+h
        [1, 0, 0, 0],    # e
        [0, 1, 0, -1],   # f-h
        [-1, 0, 1, 0],   # g-e
        [0, 0, 0, 1],    # h
        [1, 1, 0, 0],    # e+f
        [0, 0, 1, 1],    # g+h
    ], dtype=np.int64)
    
    # Reconstruction:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=np.int64)
    
    r = 0
    # m-blocks: 0, 1 (each of size 2)
    # p-blocks: 0, 1, 2, 3 (each of size 2)
    # Strassen steps: 0..6
    
    for m_block in range(2):
        for p_block in range(4):
            for s in range(7):
                # Fill U
                # A is 2x4. Index in U is i*m + j.
                # j is in [m_block*2, m_block*2+1]
                # i is in [0, 1]
                for i in range(2):
                    for jp in range(2):
                        j = m_block * 2 + jp
                        a_idx = i * m + j
                        U[a_idx, r] = X_forms[s, i * 2 + jp]
                        
                # Fill V
                # B is 4x8. Index in V is j*p + k.
                # j is in [m_block*2, m_block*2+1]
                # k is in [p_block*2, p_block*2+1]
                for jp in range(2):
                    j = m_block * 2 + jp
                    for kp in range(2):
                        k = p_block * 2 + kp
                        b_idx = j * p + k
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]
                        
                # Fill W
                # C is 2x8. Index in W is k*n + i (Trace convention W[k*n+i, r])
                # i is in [0, 1]
                # k is in [p_block*2, p_block*2+1]
                for i in range(2):
                    for kp in range(2):
                        k = p_block * 2 + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]
                        
                r += 1

    # Quick verification
    A = np.random.randint(-5, 5, size=(n, m)).astype(np.int64)
    B = np.random.randint(-5, 5, size=(m, p)).astype(np.int64)
    C_expected = A @ B
    
    A_vec = A.flatten()
    B_vec = B.flatten()
    
    C_computed = np.zeros((n, p), dtype=np.float64)
    for r_idx in range(rank):
        u_val = np.dot(U[:, r_idx], A_vec)
        v_val = np.dot(V[:, r_idx], B_vec)
        w_vec = W[:, r_idx].reshape(p, n).T
        C_computed += u_val * v_val * w_vec
        
    if not np.allclose(C_computed, C_expected):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
