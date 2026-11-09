def construct():
    n, m, p = 2, 4, 8
    rank = 1
    U = [[1]] * (n * m)  # 8 rows, 1 column
    V = [[1]] * (m * p)  # 32 rows, 1 column
    W = [[1]] * (p * n)  # 16 rows, 1 column
    return {"U": U, "V": V, "W": W}
