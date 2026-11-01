def try_rank_27(n, m, p):
    U28, V28, W28 = build_rank_28(n, m, p)
    U = np.array(U28, dtype=float)
    V = np.array(V28, dtype=float)
    W = np.array(W28, dtype=float)
    rank28 = 28
    
    # Build matrix A where A[xyz, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    A = np.zeros((n * m * p, rank28))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank28):
                    A[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                idx += 1
    
    # Find null space
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = 1e-10
    null_space = Vt[s < tol]
    
    if len(null_space) > 0:
        c = null_space[0]
        # Find index with largest |c|
        max_idx = np.argmax(np.abs(c))
        if abs(c[max_idx]) > 1e-10:
            # Construct rank 27
            U27 = np.zeros((n*m, rank28-1))
            V27 = np.zeros((m*p, rank28-1))
            W27 = np.zeros((p*n, rank28-1))
            
            # Map old indices to new
            new_col = 0
            for r in range(rank28):
                if r == max_idx:
                    continue
                scale = 1.0 - c[r] / c[max_idx]
                U27[:, new_col] = scale * U[:, r]
                V27[:, new_col] = scale * V[:, r]
                W27[:, new_col] = scale * W[:, r]
                new_col += 1
            
            return U27, V27, W27
            
    return None
