    def construct():
        n, m, p = DIMENSIONS
        # Rank 1 decomposition of the all-ones tensor
        U = [[1] * 1 for _ in range(n * m)]
        V = [[1] * 1 for _ in range(m * p)]
        W = [[1] * 1 for _ in range(p * n)]
        return {"U": U, "V": V, "W": W}
    