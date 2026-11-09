def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def validate(U, V, W):
        rank = len(U[0]) if U else 0
        T_check = np.zeros((n * m, m * p, p * n), dtype=float)
        for s in range(rank):
            u = np.array(U, dtype=float)[:, s]
            v = np.array(V, dtype=float)[:, s]
            w = np.array(W, dtype=float)[:, s]
            T_check += np.einsum('a,b,c->abc', u, v, w)
        return np.allclose(T_check, T, atol=1e-6)
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def systematic_greedy():
        T_rem = T.copy().astype(float)
        U_cols = []
        V_cols = []
        W_cols = []
        
        for step in range(n * m * p):
            if time.time() > deadline - 10:
                break
            if np.max(np.abs(T_rem)) < 1e-10:
                break
            
            best_score = -1
            best_term = None
            
            # Type 2: single (i,j), two k's
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            b1 = j * p + k1
                            b2 = j * p + k2
                            c1 = k1 * n + i
                            c2 = k2 * n + i
                            score = T_rem[a, b1, c1] + T_rem[a, b2, c2]
                            if score > best_score:
                                best_score = score
                                best_term = ('type2', i, j, k1, k2)
            
            # Type 3: two i's, single (j,k)
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    for i1 in range(n):
                        for i2 in range(i1 + 1, n):
                            a1 = i1 * m + j
                            a2 = i2 * m + j
                            c1 = k * n + i1
                            c2 = k * n + i2
                            score = T_rem[a1, b, c1] + T_rem[a2, b, c2]
                            if score > best_score:
                                best_score = score
                                best_term = ('type3', i1, i2, j, k)
            
            # Type 1: two i's, single j, two k's (cross-paired)
            for j in range(m):
                for i1 in range(n):
                    for i2 in range(i1 + 1, n):
                        for k1 in range(p):
                            for k2 in range(k1 + 1, p):
                                a1 = i1 * m + j
                                a2 = i2 * m + j
                                b1 = j * p + k1
                                b2 = j * p + k2
                                c1 = k1 * n + i1
                                c2 = k2 * n + i2
                                score = T_rem[a1, b1, c1] + T_rem[a2, b2, c2]
                                if score > best_score:
                                    best_score = score
                                    best_term = ('type1', i1, i2, j, k1, k2)
            
            if best_term is None or best_score <= 0:
                break
            
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            if best_term[0] == 'type2':
                _, i, j, k1, k2 = best_term
                u[i * m + j] = 1
                v[j * p + k1] = 1
                v[j * p + k2] = 1
                w[k1 * n + i] = 1
                w[k2 * n + i] = 1
            elif best_term[0] == 'type3':
                _, i1, i2, j, k = best_term
                u[i1 * m + j] = 1
                u[i2 * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i1] = 1
                w[k * n + i2] = 1
            elif best_term[0] == 'type1':
                _, i1, i2, j, k1, k2 = best_term
                u[i1 * m + j] = 1
                u[i2 * m + j] = 1
                v[j * p + k1] = 1
                v[j * p + k2] = 1
                w[k1 * n + i1] = 1
                w[k2 * n + i2] = 1
            
            T_rem -= np.einsum('a,b,c->abc', u, v, w)
            
            U_cols.append(u.tolist())
            V_cols.append(v.tolist())
            W_cols.append(w.tolist())
        
        if not U_cols:
            return None
        
        rank = len(U_cols)
        U = [[U_cols[s][a] for s in range(rank)] for a in range(n * m)]
        V = [[V_cols[s][b] for s in range(rank)] for b in range(m * p)]
        W = [[W_cols[s][c] for s in range(rank)] for c in range(p * n)]
        
        if validate(U, V, W):
            return U, V, W, rank
        
        return None
    
    result = systematic_greedy()
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
