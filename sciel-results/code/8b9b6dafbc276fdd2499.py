def construct():
    n, m, p = 6, 6, 6
    U = np.ones((n * m, 1), dtype=float)
    V = np.ones((m * p, 1), dtype=float)
    W = np.ones((p * n, 1), dtype=float)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
