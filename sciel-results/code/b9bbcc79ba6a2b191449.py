def construct():
    n, m, p = 3, 3, 8
    rank = 1
    U = [[1] for _ in range(n * m)]  # shape (9, 1)
    V = [[1] for _ in range(m * p)]  # shape (24, 1)  
    W = [[1] for _ in range(p * n)]  # shape (24, 1)
    return {"U": U, "V": V, "W": W}
