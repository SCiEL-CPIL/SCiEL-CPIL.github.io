    def compute_phi(N):
        phi = np.arange(N + 1, dtype=np.int64)
        for p in range(2, int(N**0.5) + 1):
            if phi[p] == p:
                # Vectorized update
                phi[p::p] -= phi[p::p] // p
        return phi
