import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Strassen-based decomposition: rank = 36
    rank_strassen = 36
    U = [[0] * rank_strassen for _ in range(n * m)]
    V = [[0] * rank_strassen for _ in range(m * p)]
    W = [[0] * rank_strassen for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 terms
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Standard 2x2x1 terms
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                # 2x2x2 Strassen block
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W[gk*n + gi][col] += val_w
                    col += 1
            else:
                # 2x2x1 standard block
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank_strassen and verify(U, V, W, rank_strassen):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
