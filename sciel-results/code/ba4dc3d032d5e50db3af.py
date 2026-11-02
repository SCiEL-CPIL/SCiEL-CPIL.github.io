def construct():
    n, m, p = DIMENSIONS
    # The target tensor is the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is 1.
    # U is (n*m, 1), V is (m*p, 1), W is (p*n, 1).
    # All entries are 1.
    rank = 1
    U = [[1.0] for _ in range(n * m)]
    V = [[1.0] for _ in range(m * p)]
    W = [[1.0] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
