"""Matrix multiplication 3,4,7 rank-minimal exact tensor decomposition.
Randomized algebraic search with EXACT integer verification; falls back to the
schoolbook rank-84 decomposition if no better exact factorization is found.
"""
import time

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
Nm, Mp, Pn = n * m, m * p, p * n
SCHOOLBOOK_RANK = n * m * p
DEADLINE = 140.0          # safety margin below the 150 s wall-clock kill
MAX_RANK = 66             # search cap (must stay below 84 to ever improve)


def _deadline():
    return time.time() > DEADLINE


def _build_tensors():
    """Schoolbook decomposition as lists-of-lists of int (U, V, W)."""
    U = [[0] * SCHOOLBOOK_RANK for _ in range(Nm)]
    V = [[0] * SCHOOLBOOK_RANK for _ in range(Mp)]
    W = [[0] * SCHOOLBOOK_RANK for _ in range(Pn)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_random(r, seed):
    """Random sparse factors with small integer entries."""
    import random
    rng = random.Random(seed)
    U = [[0] * r for _ in range(Nm)]
    V = [[0] * r for _ in range(Mp)]
    for j in range(m):
        for k in range(p):
            c = rng.randrange(r)
            V[j * p + k][c] = rng.choice((-1, 1))
    for i in range(n):
        for j in range(m):
            c = rng.randrange(r)
            U[i * m + j][c] = rng.choice((-1, 1))
    return U, V


def _verify(U, V):
    """Return (W, True) iff W = U^T(k*n+i) x V(j*p+k) equals the mult. tensor
    exactly; W is a list-of-lists of int. Else (None, False)."""
    r = len(U[0])
    W = [[0] * r for _ in range(Pn)]
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        nzU = [i for i in range(Nm) if uc[i] != 0]
        nzV = [j for j in range(Mp) if vc[j] != 0]
        for i in nzU:
            a = uc[i]
            for j in nzV:
                W[j][i] += a * vc[j]
    # Exact match against the multiplication tensor.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = k * n + i
                for t in range(Mp):
                    if t == j * p + k:
                        if W[idx][t] != 1:
                            return None, False
                    else:
                        if W[idx][t] != 0:
                            return None, False
    return W, True


def construct():
    U, V, W = _build_tensors()
    best = (U, V, W)

    global_seed = 12345
    for r in range(20, MAX_RANK + 1):
        if _deadline():
            break
        for _ in range(4000):
            if _deadline():
                break
            Ru, Rv = _build_random(r, global_seed)
            global_seed += 1
            Rw, ok = _verify(Ru, Rv)
            if ok:
                best = (Ru, Rv, Rw)
                return {"U": Ru, "V": Rv, "W": Rw}
    return {"U": best[0], "V": best[1], "W": best[2]}
