def construct():
    import numpy as np
    import time
    
    deadline = time.time() + 140
    
    n, m, p = 2, 4, 5
    
    # Find Strassen signs for 2x2x2
    np.random.seed(42)
    Ms = []
    Cs = []
    for trial in range(20):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C = A @ B
        a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
        e, f, g, h = B[0,0], B[0,1], B[1,0], B[1,1]
        m1 = (a+d)*(e+h)
        m2 = (c+d)*e
        m3 = a*(f-h)
        m4 = d*(g-e)
        m5 = (a+b)*h
        m6 = (c-a)*(e+f)
        m7 = (b-d)*(g+h)
        Ms.append([m1, m2, m3, m4, m5, m6, m7])
        Cs.append([C[0,0], C[0,1], C[1,0], C[1,1]])
    
    Ms = np.array(Ms)
    Cs = np.array(Cs)
    
    signs = {}
    valid = True
    for idx, name in enumerate(['C00', 'C01', 'C10', 'C11']):
        s, _, _, _ = np.linalg.lstsq(Ms, Cs[:, idx], rcond=None)
        s_rounded = np.clip(np.round(s), -1, 1)
        error = np.max(np.abs(Ms @ s_rounded - Cs[:, idx]))
        if error > 1e-10:
            valid = False
            break
        signs[name] = s_rounded
    
    if not valid:
        # Fall back to schoolbook
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build rank-36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    s00, s01, s10, s11 = signs['C00'], signs['C01'], signs['C10'], signs['C11']
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    def add_strassen(i0, i1, ka, kb, jc, jd):
        nonlocal col
        for r in range(7):
            u_rows = [i0*4+ka, i0*4+kb, i1*4+ka, i1*4+kb]
            for idx, val in enumerate(strassen_U[r]):
                if val != 0:
                    U[u_rows[idx]][col] += val
            v_rows = [ka*5+jc, ka*5+jd, kb*5+jc, kb*5+jd]
            for idx, val in enumerate(strassen_V[r]):
                if val != 0:
                    V[v_rows[idx]][col] += val
            w_rows = [jc*2+i0, jd*2+i0, jc*2+i1, jd*2+i1]
            w_vals = [s00[r], s01[r], s10[r], s11[r]]
            for idx, val in enumerate(w_vals):
                if val != 0:
                    W[w_rows[idx]][col] += val
            col += 1
    
    def add_schoolbook(i0, i1, ka, kb, jc):
        nonlocal col
        for i in [i0, i1]:
            for k in [ka, kb]:
                U[i*4+k][col] = 1
                V[k*5+jc][col] = 1
                W[jc*2+i][col] = 1
                col += 1
    
    # A1*B1: k in {0,1}, j split into {0,1}, {2,3}, {4}
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook(0, 1, 0, 1, 4)
    # A2*B2: k in {2,3}, j split into {0,1}, {2,3}, {4}
    add_strassen(0, 1, 2, 3, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3)
    add_schoolbook(0, 1, 2, 3, 4)
    
    assert col == rank
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0
    for i in range(2):
        for k in range(4):
            for j in range(5):
                val = np.sum(U_np[i*4+k, :] * V_np[k*5+j, :] * W_np[j*2+i, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fall back
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
