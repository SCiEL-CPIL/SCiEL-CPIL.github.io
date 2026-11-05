def construct():
    n, m, p = 6, 6, 6
    # Strassen coefficients
    # For each term m (0-6), define alpha[a,b] and beta[b,c]
    # a,c in {0,1}, b in {0,1}
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        {'alpha': {(0,0):1, (1,1):1}, 'beta': {(0,0):1, (1,1):1}},
        # M2: (A21+A22)B11
        {'alpha': {(1,0):1, (1,1):1}, 'beta': {(0,0):1}},
        # M3: A11(B12-B22)
        {'alpha': {(0,0):1}, 'beta': {(0,1):1, (1,1):-1}},
        # M4: A22(B21-B11)
        {'alpha': {(1,1):1}, 'beta': {(1,0):1, (0,0):-1}},
        # M5: (A11+A12)B22
        {'alpha': {(0,0):1, (0,1):1}, 'beta': {(1,1):1}},
        # M6: (A21-A11)(B11+B12)
        {'alpha': {(1,0):1, (0,0):-1}, 'beta': {(0,0):1, (0,1):1}},
        # M7: (A12-A22)(B21+B22)
        {'alpha': {(0,1):1, (1,1):-1}, 'beta': {(1,0):1, (1,1):1}},
    ]
    
    # For each Strassen term, for each (a,b) in alpha, (b,c) in beta,
    # and for each r,s,t in 0..2, create a rank-1 tensor
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    for term in strassen_terms:
        for (a,b), alpha_val in term['alpha'].items():
            for (b2,c), beta_val in term['beta'].items():
                if b != b2:
                    continue  # b must match
                for r in range(3):
                    for s in range(3):
                        for t in range(3):
                            i = a*3 + r
                            j = b*3 + s
                            k = c*3 + t
                            U_rows.append((i*m+j, alpha_val))
                            V_rows.append((j*p+k, beta_val))
                            W_rows.append((k*n+i, 1))
                            rank += 1
    
    # Build U, V, W matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        u_idx, u_val = U_rows[c]
        v_idx, v_val = V_rows[c]
        w_idx, w_val = W_rows[c]
        U[u_idx][c] = u_val
        V[v_idx][c] = v_val
        W[w_idx][c] = w_val
    
    return {"U": U, "V": V, "W": W}
