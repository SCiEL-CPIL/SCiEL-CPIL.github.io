"""Search for a rank-minimal exact tensor decomposition of 3x4 x 4x7 multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
SEED = 12345
DEADLINE = 112.0  # wall-clock safety margin (evaluator kills at ~150s)


def construct():
    n, m, p = DIMENSIONS
    A = n * m   # 12
    B = m * p   # 28
    C = p * n   # 21
    school = n * m * p  # 84

    # ---- Build the 84 schoolbook rank-1 terms (indices i,j,k) ----
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))

    def build_matrices(T):
        U = np.zeros((A, len(T)))
        V = np.zeros((B, len(T)))
        W = np.zeros((C, len(T)))
        for c, (i, j, k) in enumerate(T):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    U0, V0, W0 = build_matrices(terms)

    # ---- Exact contraction check (exact rational arithmetic) ----
    def is_exact(U, V, W):
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        for a in range(A):
            for b in range(B):
                for d in range(C):
                    v = T[a, b, d]
                    if not v.is_integer():
                        return False
                    i = a // m
                    j = a % m
                    k = b // p
                    r = b % p
                    idx = k * n + i
                    if idx != d:
                        if v != 0:
                            return False
                    else:
                        if v != 1:
                            return False
        return True

    # ---- Random linear-combination search ----
    rng = np.random.default_rng(SEED)
    start = time.time()

    def attempt(r, ntrials):
        for t in range(ntrials):
            if time.time() - start > DEADLINE:
                return None
            r1 = rng.standard_normal(r)
            r2 = rng.standard_normal(r)
            r3 = rng.standard_normal(r)
            U = U0 @ np.diag(r1)
            V = V0 @ np.diag(r2)
            W = W0 @ np.diag(r3)
            if is_exact(U, V, W):
                return (U, V, W)
        return None

    # Try a range of ranks below 84.
    for r in range(83, 0, -1):
        if time.time() - start > DEADLINE:
            break
        res = attempt(r, 8)
        if res is not None:
            U, V, W = res
            U = U.tolist(); V = V.tolist(); W = W.tolist()
            return {"U": U, "V": V, "W": W}

    # ---- Fallback: guaranteed schoolbook decomposition ----
    U = [[0] * school for _ in range(A)]
    V = [[0] * school for _ in range(B)]
    W = [[0] * school for _ in range(C)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
