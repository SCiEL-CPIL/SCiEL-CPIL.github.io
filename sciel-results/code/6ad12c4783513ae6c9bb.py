<<<<<<< SEARCH
def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    # ALS iterations
    for it in range(50):
        if time.time() > deadline - 3:
            break

        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2

    if check(U43, V43, W43, n, m, p):
        return U43, V43, W43

    # Local search: try flipping individual half-integer values
    for ls_iter in range(200):
        if time.time() > deadline - 1:
            break
        
        improved = False
        # Try flipping U entries
        for a in range(n * m):
            for r in range(R):
                old = U43[a, r]
                for delta in [0.5, -0.5]:
                    U43[a, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    U43[a, r] = old
        
        # Try flipping V entries
        for b in range(m * p):
            for r in range(R):
                old = V43[b, r]
                for delta in [0.5, -0.5]:
                    V43[b, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    V43[b, r] = old
        
        # Try flipping W entries
        for c in range(p * n):
            for r in range(R):
                old = W43[c, r]
                for delta in [0.5, -0.5]:
                    W43[c, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    W43[c, r] = old
    
    return None
=======
def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    # Pre-compute index arrays for vectorized ALS
    # For U[i*m+j]: M[k, r] = V[j*p+k, r] * W[k*n+i, r]
    # For V[j*p+k]: M[i, r] = U[i*m+j, r] * W[k*n+i, r]
    # For W[k*n+i]: M[j, r] = U[i*m+j, r] * V[j*p+k, r]

    # ALS iterations
    for it in range(80):
        if time.time() > deadline - 3:
            break

        # Update U: for each (i,j), solve M.T @ x = ones(p)
        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                # M is p x R
                M = V43[j*p:(j+1)*p, :] * W43[0:p*n:n, :][:p, :]  # W[k*n+i] for k=0..p-1
                # Actually W43[k*n+i] for k in 0..p-1
                W_col = np.array([W43[k*n+i, :] for k in range(p)])
                M = V43[j*p:(j+1)*p, :] * W_col
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i*m+j, :] = sol
                except:
                    pass

        # Update V: for each (j,k), solve M.T @ x = ones(n)
        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                U_col = np.array([U43[i*m+j, :] for i in range(n)])
                W_col = np.array([W43[k*n+i, :] for i in range(n)])
                M = U_col * W_col
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j*p+k, :] = sol
                except:
                    pass

        # Update W: for each (k,i), solve M.T @ x = ones(m)
        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                U_row = U43[i*m:(i+1)*m, :]
                V_col = V43[k:(k+1)*p:p, :]  # V[j*p+k] for j=0..m-1
                M = U_row * V_col
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k*n+i, :] = sol
                except:
                    pass

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2

    if check(U43, V43, W43, n, m, p):
        return U43, V43, W43

    # Local search: try flipping individual half-integer values
    for ls_iter in range(300):
        if time.time() > deadline - 1:
            break
        
        # Try flipping U entries
        for a in range(n * m):
            for r in range(R):
                old = U43[a, r]
                for delta in [0.5, -0.5]:
                    U43[a, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    U43[a, r] = old
        
        # Try flipping V entries
        for b in range(m * p):
            for r in range(R):
                old = V43[b, r]
                for delta in [0.5, -0.5]:
                    V43[b, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    V43[b, r] = old
        
        # Try flipping W entries
        for c in range(p * n):
            for r in range(R):
                old = W43[c, r]
                for delta in [0.5, -0.5]:
                    W43[c, r] = old + delta
                    if check(U43, V43, W43, n, m, p):
                        return U43, V43, W43
                    W43[c, r] = old
    
    return None
>>>>>>> REPLACE

Now let me also improve the `construct` function to try columns in a smarter order (columns that are more "redundant" or have smaller norms first):

