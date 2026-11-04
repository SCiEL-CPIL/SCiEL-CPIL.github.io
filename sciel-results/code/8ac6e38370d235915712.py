def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
    
    best = (rank_school, U_s, V_s, W_s)
    
    # Try various ranks with ALS
    for target_rank in [51, 48, 45, 42, 40, 38, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21]:
        if time.time() > deadline - 15:
            break
        
        success = False
        for seed in range(3):
            if time.time() > deadline - 8:
                break
            try:
                U, V, W = _als_cp_decomposition(np.ones((n,m,p)), target_rank, n_iter=200, seed=seed)
                # Try rounding
                for denom in [1, 2, 4, 8]:
                    Ur = np.round(U*denom)/denom
                    Vr = np.round(V*denom)/denom
                    Wr = np.round(W*denom)/denom
                    if _verify_decomposition(Ur, Vr, Wr, n, m, p, tol=1e-10):
                        if target_rank < best[0]:
                            best = (target_rank, Ur, Vr, Wr)
                        success = True
                        break
                if success:
                    break
            except:
                continue
    
    r, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
