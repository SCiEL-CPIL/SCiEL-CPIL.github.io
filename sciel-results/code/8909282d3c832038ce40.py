import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # We'll build the decomposition term by term
    terms = []  # each term is (u, v, w) where u is length n*m=16, v is length m*p=28, w is length p*n=28
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Schoolbook for simplicity - this gives rank 112
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
