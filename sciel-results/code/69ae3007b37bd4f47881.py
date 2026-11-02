def construct():
    n, m, p = 3, 3, 6
    rank = 1
    U = [[1] for _ in range(n * m)]  # 9x1
    V = [[1] for _ in range(m * p)]  # 18x1
    W = [[1] for _ in range(p * n)]  # 18x1
    return {"U": U, "V": V, "W": W}
