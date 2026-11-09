def construct():
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    rank = 1
    U = [[1.0] for _ in range(nm)]
    V = [[1.0] for _ in range(mp)]
    W = [[1.0] for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
