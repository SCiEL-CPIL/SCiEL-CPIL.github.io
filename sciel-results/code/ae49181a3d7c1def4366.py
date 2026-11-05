"""Exact rank-minimal decomposition search for 3x5 * 5x6 (Strassen block attempt + schoolbook fallback)."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(n, m, p, U, V, W):
    import numpy as np
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    T = np.einsum('ijr,jkr->ijk', Ua, Va)
    Wt = np.einsum('kir->ki', Wa)
    if T.shape != (n, m, p) or Wt.shape != (p, n):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expect = 1 if (Wt[k, i] == 1) else 0
                if T[i, j, k] != expect:
                    return False
    return True


def _build_via_verified_mapping(n, m, p):
    a = m // 2
    b = m - a
    def Ui(i, j): return i * m + j
    def Vi(j, k): return j * p + k
    def Wi(k, i): return k * n + i

    def c00_terms(A00, A01, A10, A11, B00, B01, B10, B11):
        terms = []
        terms += [(A00,B10,1),(A00,B11,1),(A11,B10,1),(A11,B11,1)]
        terms += [(A11,B10,1),(A11,B00,-1)]
        terms += [(A00,B11,-1),(A01,B11,-1)]
        terms += [(A01,B10,1),(A01,B11,1),(A11,B10,-1),(A11,B11,-1)]
        return terms

    A_ranges = {'A0': (0, a), 'A1': (a, m)}
    B_ranges = {'B0': (0, a), 'B1': (a, m)}

    mappings = [
        ('A0','A1','A1','A0','B0','B1','B1','B0'),
        ('A0','A0','A1','A1','B0','B1','B0','B1'),
        ('A0','A1','A0','A1','B0','B0','B1','B1'),
        ('A0','A1','A1','A0','B0','B0','B1','B1'),
        ('A0','A0','A0','A1','B0','B1','B0','B1'),
        ('A0','A1','A0','A0','B0','B1','B1','B0'),
        ('A0','A1','A1','A0','B0','B1','B0','B1'),
        ('A0','A0','A1','A0','B0','B1','B1','B0'),
    ]

    for mp in mappings:
        A00, A01, A10, A11, B00, B01, B10, B11 = mp
        terms = c00_terms(A00, A01, A10, A11, B00, B01, B10, B11)
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r, (Ablk, Bblk, coeff) in enumerate(terms):
            ja, jb = A_ranges[Ablk]
            jk0, jk1 = B_ranges[Bblk]
            for i in range(n):
                for j in range(ja, jb):
                    U[Ui(i, j)][r] = coeff
            for j in range(jk0, jk1):
                for k in range(p):
                    V[Vi(j, k)][r] = coeff
            for k in range(p):
                for i in range(n):
                    W[Wi(k, i)][r] = 1
        if _verify(n, m, p, U, V, W):
            return U, V, W
    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    import time
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0
    n, m, p = DIMENSIONS

    if time.time() < DEADLINE - MARGIN:
        res = _build_via_verified_mapping(n, m, p)
        if res is not None:
            U, V, W = res
            if _verify(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}

    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
