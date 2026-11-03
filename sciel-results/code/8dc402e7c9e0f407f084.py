"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    import random
    rng = random.Random(42)
    full_check = rank <= 60
    if full_check:
        checks = [(a, b, cidx) for a in range(n*m) for b in range(m*p) for cidx in range(p*n)
                  if not (a % m == b // p and a // m == cidx % n and b % p == cidx // n)]
    else:
        checks = [(rng.randrange(n*m), rng.randrange(m*p), rng.randrange(p*n)) for _ in range(400)]
    for a, b, cidx in checks:
        s = 0
        for c in range(rank):
            s += U[a][c] * V[b][c] * W[cidx][c]
        if s != 0:
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    base = _schoolbook(n, m, p)
    return base
# EVOLVE-BLOCK-END

