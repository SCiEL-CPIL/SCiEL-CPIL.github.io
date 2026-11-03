def construct():
    n, m, p = 2, 3, 9
    U = [[1.0]] * (n * m)  # 6 x 1
    V = [[1.0]] * (m * p)  # 27 x 1
    W = [[1.0]] * (p * n)  # 18 x 1
    return {"U": U, "V": V, "W": W}
