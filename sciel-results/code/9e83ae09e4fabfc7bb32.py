    def construct():
        n, m, p = DIMENSIONS
        # Rank 1 decomposition
        R = 1
        U = np.ones((n * m, R))
        V = np.ones((m * p, R))
        W = np.ones((p * n, R))
        
        # Verify
        # ... (optional, but good for safety)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    