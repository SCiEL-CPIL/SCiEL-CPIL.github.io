def construct():
    n, m, p = 2, 4, 4
    R = 1
    U = np.ones((n * m, R), dtype=int)
    V = np.ones((m * p, R), dtype=int)
    W = np.ones((p * n, R), dtype=int)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
