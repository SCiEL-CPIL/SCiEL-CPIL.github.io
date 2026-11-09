def construct():
    n, m, p = DIMENSIONS
    # All-ones tensor has rank 1
    U = np.ones((n*m, 1))
    V = np.ones((m*p, 1))
    W = np.ones((p*n, 1))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
