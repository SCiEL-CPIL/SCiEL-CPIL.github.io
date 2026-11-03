    def verify(U, V, W):
        """Check if decomposition is valid using einsum with integers."""
        r = U.shape[1]
        U_idx = U.reshape(n, m, r)
        V_idx = V.reshape(m, p, r)
        W_idx = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        return np.array_equal(T, np.ones((n, m, p), dtype=U.dtype))
