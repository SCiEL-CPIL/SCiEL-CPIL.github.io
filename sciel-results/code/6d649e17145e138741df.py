"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
_DEADLINE = 100.0  # safety margin before 120s kill

def _build_groups(total, prefer2):
    """Partition range(total) into groups of size 2 (as many as possible) then a leftover of size 1."""
    groups = []
    i = 0
    while i + 2 <= total:
        groups.append([i, i + 1])
        i += 2
    if i < total:
        groups.append([i])
    return groups

def _strassen_terms(nrows, mcols, pcols, n, m, p):
    """Return list of (U, V, W) sparse dicts for a 2x2x2 Strassen block.
    nrows: list of 2 row indices of A (and C).
    mcols: list of 2 col indices of A (and row indices of B).
    pcols: list of 2 col indices of B (and C).
    """
    r0, r1 = nrows
    c0, c1 = mcols
    d0, d1 = pcols
    # A indices: (r, c) -> r*m + c
    # B indices: (c, d) -> c*p + d
    # C indices: (r, d) -> d*n + r   (trace convention W[k*n+i])
    terms = []
    # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    terms.append((
        {(r0, c0): 1, (r1, c1): 1},
        {(c0, d0): 1, (c1, d1): 1},
        {(r0, d0): 1, (r1, d1): 1},
    ))
    # M2 = (A21+A22)(B11) -> C21 += 1, C22 -= 1
    terms.append((
        {(r1, c0): 1, (r1, c1): 1},
        {(c0, d0): 1},
        {(r1, d0): 1, (r1, d1): -1},
    ))
    # M3 = (A11)(B12-B22) -> C12 += 1, C22 += 1
    terms.append((
        {(r0, c0): 1},
        {(c0, d1): 1, (c1, d1): -1},
        {(r0, d1): 1, (r1, d1): 1},
    ))
    # M4 = (A22)(B21-B11) -> C11 += 1, C21 += 1
    terms.append((
        {(r1, c1): 1},
        {(c1, d0): 1, (c0, d0): -1},
        {(r0, d0): 1, (r1, d0): 1},
    ))
    # M5 = (A11+A12)(B22) -> C11 -= 1, C12 += 1
    terms.append((
        {(r0, c0): 1, (r0, c1): 1},
        {(c1, d1): 1},
        {(r0, d0): -1, (r0, d1): 1},
    ))
    # M6 = (A21-A11)(B11+B12) -> C22 += 1
    terms.append((
        {(r1, c0): 1, (r0, c0): -1},
        {(c0, d0): 1, (c0, d1): 1},
        {(r1, d1): 1},
    ))
    # M7 = (A12-A22)(B21+B22) -> C11 += 1
    terms.append((
        {(r0, c1): 1, (r1, c1): -1},
        {(c1, d0): 1, (c1, d1): 1},
        {(r0, d0): 1},
    ))
    return terms

def _build_terms(n, m, p):
    ngroups = _build_groups(n, 2)
    mgroups = _build_groups(m, 2)
    pgroups = _build_groups(p, 2)
    terms = []
    for ng in ngroups:
        for mg in mgroups:
            for pg in pgroups:
                if len(ng) == 2 and len(mg) == 2 and len(pg) == 2:
                    terms.extend(_strassen_terms(ng, mg, pg, n, m, p))
                else:
                    for i in ng:
                        for j in mg:
                            for k in pg:
                                terms.append((
                                    {(i, j): 1},
                                    {(j, k): 1},
                                    {(i, k): 1},
                                ))
    return terms

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, (ud, vd, wd) in enumerate(terms):
        for (i, j), v in ud.items():
            U[i * m + j][col] = v
        for (j, k), v in vd.items():
            V[j * p + k][col] = v
        for (i, k), v in wd.items():
            W[k * n + i][col] = v
    return U, V, W

def _verify(U, V, W, n, m, p, trials=4, seed=0):
    rng = np.random.default_rng(seed)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = rng.normal(size=(n, m))
        B = rng.normal(size=(m, p))
        Af = A.reshape(-1)  # index i*m+j
        Bf = B.reshape(-1)  # index j*p+k
        # C = sum_col (Ua[:,col]·Af) * (Va[:,col]·Bf) * Wa[:,col]
        C = np.zeros((p, n))  # W index k*n+i
        for col in range(Ua.shape[1]):
            a = float(Ua[:, col] @ Af)
            b = float(Va[:, col] @ Bf)
            C += (a * b) * Wa[:, col]
        C = C.reshape((p, n)).T  # to (n, p): C[i,k]
        expected = A @ B
        if not np.allclose(C, expected, atol=1e-8):
            return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Try the block-Strassen scheme (rank 99).
    try:
        if time.time() - start < _DEADLINE:
            terms = _build_terms(n, m, p)
            U, V, W = _terms_to_matrices(terms, n, m, p)
            if time.time() - start < _DEADLINE:
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    # Fallback: schoolbook (rank n*m*p = 105).
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
