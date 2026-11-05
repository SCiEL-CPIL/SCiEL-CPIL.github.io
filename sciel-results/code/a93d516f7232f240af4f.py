def _verify(U, V, W, n=2, m=3, p=5, tol=1e-9):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We need to check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Construct T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                u_val = U[i*m+j, :]
                v_val = V[j*p+k, :]
                w_val = W[k*n+i, :]
                T[i, j, k] = np.dot(u_val, v_val * w_val)
                
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)
