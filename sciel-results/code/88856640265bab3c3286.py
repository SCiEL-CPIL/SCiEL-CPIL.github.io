def find_rank(T, r, max_iter=200, seed=0, deadline=None):
    I, J, K = T.shape
    rng = np.random.RandomState(seed)
    best = None
    for attempt in range(3):
        U = rng.randn(I, r)
        V = rng.randn(J, r)
        W = rng.randn(K, r)
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            # Update U
            M = np.einsum('jr,kr->rjk', V, W).reshape(r, J*K)
            Tflat = T.reshape(I, J*K)
            U, *_ = np.linalg.lstsq(M.T, Tflat.T, rcond=None)
            U = U.T
            # Update V
            M = np.einsum('ir,kr->rik', U, W).reshape(r, I*K)
            Tflat = T.transpose(1,0,2).reshape(J, I*K)
            V, *_ = np.linalg.lstsq(M.T, Tflat.T, rcond=None)
            V = V.T
            # Update W
            M = np.einsum('ir,jr->rij', U, V).reshape(r, I*J)
            Tflat = T.transpose(2,0,1).reshape(K, I*J)
            W, *_ = np.linalg.lstsq(M.T, Tflat.T, rcond=None)
            W = W.T
            if deadline is not None and time.time() > deadline:
                break
        # Round to integer or half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, T, tol=1e-6):
            return U_r, V_r, W_r
    return None
