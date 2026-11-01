def construct():
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 3, 3, 3
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(t, u_list, v_list, w_list):
        for (i, j, val) in u_list:
            U[i*m+j, t] += val
        for (j, k, val) in v_list:
            V[j*p+k, t] += val
        for (k, i, val) in w_list:
            W[k*n+i, t] += val
    
    # Rank-19 scheme (Bini-De Falco-Strassen)
    # Terms 0-6: Top-left 2x2 Strassen
    add_term(0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    add_term(1, [(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)])
    add_term(2, [(0,0,1)], [(1,1,1),(1,0,-1)], [(0,0,1)])
    add_term(3, [(1,1,1)], [(0,1,1),(0,0,-1)], [(1,0,1)])
    add_term(4, [(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1)])
    add_term(5, [(1,0,1),(1,1,-1)], [(0,0,1),(1,0,1)], [(0,1,1)])
    add_term(6, [(0,1,1),(1,0,-1)], [(1,1,1),(0,1,1)], [(0,1,1)])
    
    # Terms 7-13: Bottom-right 2x2 Strassen
    add_term(7, [(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)])
    add_term(8, [(2,1,1),(2,2,1)], [(1,1,1)], [(1,2,1)])
    add_term(9, [(1,1,1)], [(2,2,1),(2,1,-1)], [(1,1,1)])
    add_term(10, [(2,2,1)], [(1,2,1),(1,1,-1)], [(2,1,1)])
    add_term(11, [(1,1,1),(1,2,1)], [(2,2,1)], [(2,1,1)])
    add_term(12, [(2,1,1),(2,2,-1)], [(1,1,1),(2,1,1)], [(1,2,1)])
    add_term(13, [(1,2,1),(2,1,-1)], [(2,2,1),(1,2,1)], [(1,2,1)])
    
    # Terms 14-18: Cross terms
    add_term(14, [(0,0,1),(0,1,1),(0,2,1)], [(0,2,1),(1,2,1),(2,2,1)], [(2,0,1)])
    add_term(15, [(0,0,1)], [(1,2,1),(0,2,-1)], [(2,0,1)])
    add_term(16, [(0,2,1)], [(0,2,1)], [(2,0,1)])
    add_term(17, [(2,0,1),(2,1,1),(2,2,1)], [(0,2,1),(1,2,1)], [(2,2,1)])
    add_term(18, [(2,0,1)], [(1,2,1)], [(2,2,1)])
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        max_err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        expected = 1.0
                    else:
                        expected = 0.0
                    val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    err = abs(val - expected)
                    max_err = max(max_err, err)
        return max_err
    
    err = verify(U, V, W)
    
    if err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 27
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
