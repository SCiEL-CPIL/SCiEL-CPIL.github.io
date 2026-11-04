def construct():
    n, m, p = DIMENSIONS
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    return {"U": U, "V": W, "W": W}  # rank 1
