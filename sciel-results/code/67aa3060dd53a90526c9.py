    def _try_rank16_random(deadline):
        n, m, p = DIMENSIONS
        n_a, n_b, n_c = n*m, m*p, p*n
        T = _build_tensor(n, m, p)
        rank = 16
        Tflat = T.reshape(n_a * n_b * n_c)

        def error_of(U, V, W):
            res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            return np.sum((res.reshape(-1) - Tflat) ** 2)

        def refine(U, V, W, max_sweeps, deadline):
            for sweep in range(max_sweeps):
                if time.time() > deadline:
                    break
                # Update U
                for r in range(rank):
                    v = V[:, r]; w = W[:, r]
                    res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    res -= np.einsum('a,b,c->abc', U[:, r], v, w)
                    feat = (v[:, None] * w[None, :]).reshape(-1)
                    resA = res.reshape(n_a, -1)
                    f2 = float(feat @ feat)
                    if f2 < 1e-14: continue
                    U[:, r] = -(resA @ feat) / f2
                # Update V
                for r in range(rank):
                    u = U[:, r]; w = W[:, r]
                    res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    res -= np.einsum('a,b,c->abc', u, V[:, r], w)
                    feat = (u[:, None] * w[None, :]).reshape(-1)
                    resB = res.reshape(n_b, -1)
                    f2 = float(feat @ feat)
                    if f2 < 1e-14: continue
                    V[:, r] = -(resB @ feat) / f2
                # Update W
                for r in range(rank):
                    u = U[:, r]; v = V[:, r]
                    res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    res -= np.einsum('a,b,c->abc', u, v, W[:, r])
                    feat = (u[:, None] * v[None, :]).reshape(-1)
                    resC = res.reshape(n_c, -1)
                    f2 = float(feat @ feat)
                    if f2 < 1e-14: continue
                    W[:, r] = -(resC @ feat) / f2
            return U, V, W

        best_err = np.inf
        best = None
        # Random restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(n_a, rank)
            V = np.random.randn(n_b, rank)
            W = np.random.randn(n_c, rank)
            
            # Normalize to prevent explosion
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)

            sweeps = 100
            U, V, W = refine(U, V, W, sweeps, deadline)
            err = error_of(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-6:
                return U, V, W
        
        if best is not None and best_err < 1e-6:
            return best
        return None
