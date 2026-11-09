def construct():
    n, m, p = 3, 5, 5
    # Rank 1: all ones
    U = np.ones((n*m, 1))
    V = np.ones((m*p, 1)) 
    W = np.ones((p*n, 1))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
