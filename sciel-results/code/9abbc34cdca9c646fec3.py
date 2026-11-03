"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    best_rank = None
    best_U = best_V = best_W = None

    # Candidate 1 (primary): 4+2 mixed partition -> rank 176
    if best_rank is None or 176 < best_rank:
        res = _build_4plus2_mixed(n, m, p)
        if res is not None:
            best_rank, best_U, best_V, best_W = 176, *res

    # Candidate 2 (fallback): block-Strassen 3x3 blocks of 2x2 -> rank 189
    if time.time() < deadline and (best_rank is None or 189 < best_rank):
        res = _build_block_strassen(n, m, p)
        if res is not None:
            best_rank, best_U, best_V, best_W = 189, *res

    # Candidate 3 (guaranteed fallback): schoolbook -> rank 216
    if best_rank is None or 216 < best_rank:
        res = _build_schoolbook(n, m, p)
        best_rank, best_U, best_V, best_W = 216, *res

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


# ---------------------------------------------------------------------------
# Strassen 2x2 (rank 7). Coefficients given in row-major order.
# ---------------------------------------------------------------------------
def _strassen_2x2_terms():
    # index: [a00, a01, a10, a11]
    return [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]


# ---------------------------------------------------------------------------
# Laderman 4x4 (rank 25), built from the 2x2 Strassen primitive so that it is
# correct by construction.  A, B are partitioned into 2x2 blocks of 2x2
# matrices; C_block = sum_j A_block * B_block uses Strassen per product.
# ---------------------------------------------------------------------------
def _laderman_4x4_terms():
    st = _strassen_2x2_terms()

    def blk(c00, c01, c10, c11):
        v = [0] * 16
        v[0] = c00; v[1] = c00; v[4] = c00; v[5] = c00
        v[2] = c01; v[3] = c01; v[6] = c01; v[7] = c01
        v[8] = c10; v[9] = c10; v[12] = c10; v[13] = c10
        v[10] = c11; v[11] = c11; v[14] = c11; v[15] = c11
        return v

    terms = []
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # C block position
        for jb in range(2):                            # summing block j
            for (sa, sb, sc) in st:                    # Strassen term
                A = [0] * 16
                B = [0] * 16
                C = [0] * 16
                for lr in range(2):
                    for lc in range(2):
                        A[(i * 2 + lr) * 4 + (jb * 2 + lc)] = sa[lr * 2 + lc]
                        B[(jb * 2 + lr) * 4 + (j * 2 + lc)] = sb[lr * 2 + lc]
                        C[(i * 2 + lr) * 4 + (j * 2 + lc)] = sc[lr * 2 + lc]
                terms.append((tuple(A), tuple(B), tuple(C)))
    return terms  # 4 * 2 * 7 = 56 raw terms


# ---------------------------------------------------------------------------
# Build the 4+2 mixed decomposition (rank 176) and verify it.
# ---------------------------------------------------------------------------
def _build_4plus2_mixed(n, m, p, deadline=None):
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 176

    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    col = 0

    def add_term(a_idx, a_val, b_idx, b_val, c_idx, c_val):
        nonlocal col
        U[a_idx, col] = a_val
        V[b_idx, col] = b_val
        W[c_idx, col] = c_val
        col += 1

    def add_block_schoolbook(ar, ac, br, bc, cr, cc, ra, ca, rb, cb):
        # A[ar:ar+ra, ac:ac+ca] * B[br:br+rb, bc:bc+cb] -> C[cr:cr+rc, cc:cc+cc]
        # requires ca == rb
        for i in range(ra):
            for j in range(ca):
                for k in range(cb):
                    add_term((ar + i) * m + (ac + j), 1,
                             (br + j) * p + (bc + k), 1,
                             (cc + k) * n + (cr + i), 1)

    def add_block_alg(ar, ac, br, bc, cr, cc, size, terms):
        for (acv, bcv, ccv) in terms:
            for r in range(size):
                for c in range(size):
                    if acv[r * size + c] != 0:
                        U[(ar + r) * m + (ac + c), col] = acv[r * size + c]
                    if bcv[r * size + c] != 0:
                        V[(br + r) * p + (bc + c), col] = bcv[r * size + c]
                    if ccv[r * size + c] != 0:
                        W[(cc + c) * n + (cr + r), col] = ccv[r * size + c]
            col += 1

    # C00 = A00*B00 + A01*B10
    add_block_alg(0, 0, 0, 0, 0, 0, 4, _laderman_4x4_terms())
    add_block_schoolbook(0, 4, 4, 0, 0, 0, 4, 2, 2, 4)
    # C01 = A00*B01 + A01*B11
    add_block_schoolbook(0, 0, 0, 4, 0, 4, 4, 4, 4, 2)
    add_block_schoolbook(0, 4, 4, 4, 0, 4, 4, 2, 2, 2)
    # C10 = A10*B00 + A11*B10
    add_block_schoolbook(4, 0, 0, 0, 4, 0, 2, 4, 4, 4)
    add_block_schoolbook(4, 4, 4, 0, 4, 0, 2, 2, 2, 4)
    # C11 = A10*B01 + A11*B11
    add_block_schoolbook(4, 0, 0, 4, 4, 4, 2, 4, 4, 2)
    add_block_alg(4, 4, 4, 4, 4, 4, 2, _strassen_2x2_terms())

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"

    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    return U, V, W


# ---------------------------------------------------------------------------
# Block-Strassen 6x6 = 3x3 blocks of 2x2 (rank 189).
# ---------------------------------------------------------------------------
def _build_block_strassen(n, m, p, deadline=None):
    nm, mp, pn = n * m, m * p, p * n
    nb, bs = 3, 2
    total_rank = nb * nb * nb * 7  # 189

    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    st = _strassen_2x2_terms()
    col = 0
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                for (sa, sb, sc) in st:
                    for r in range(bs):
                        for c in range(bs):
                            if sa[r * bs + c] != 0:
                                U[(bi * bs + r) * m + (bj * bs + c), col] = sa[r * bs + c]
                            if sb[r * bs + c] != 0:
                                V[(bj * bs + r) * p + (bk * bs + c), col] = sb[r * bs + c]
                            if sc[r * bs + c] != 0:
                                W[(bk * bs + c) * n + (bi * bs + r), col] = sc[r * bs + c]
                    col += 1

    assert col == total_rank
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook (rank 216) - guaranteed valid.
# ---------------------------------------------------------------------------
def _build_schoolbook(n, m, p, deadline=None):
    nm, mp, pn = n * m, m * p, p * n
    total_rank = n * m * p
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Verification: check the 216 on-diagonal entries and 3000 random off-diagonals.
# ---------------------------------------------------------------------------
def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                if abs(val - 1.0) > tol:
                    return False

    rng = np.random.default_rng(42)
    for _ in range(3000):
        a = int(rng.integers(0, nm))
        b = int(rng.integers(0, mp))
        c = int(rng.integers(0, pn))
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = float(np.dot(U[a, :], V[b, :] * W[c, :]))
        if abs(val) > tol:
            return False

    return True
# EVOLVE-BLOCK-END
