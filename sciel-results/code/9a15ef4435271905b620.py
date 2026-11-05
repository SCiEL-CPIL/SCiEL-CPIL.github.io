def construct():
    n, m, p = 2, 4, 5
    rank = 1
    U = np.ones((n*m, rank))
    V = np.ones((m*p, rank))
    W = np.ones((p*n, rank))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
