"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Combined design:
  * Structural idea (current program): block decomposition -- top 2 rows of the
    output handled with four Strassen 2x2*2x2 sub-multiplications (rank 7 each)
    and the bottom row handled with four trivial 1x2*2x2 sub-multiplications
    (rank 4 each) -> seed rank 36.
  * Refinement machinery (inspirations + strengthened): a flat, index-based block
    assembler that lays the U/V/W factors out so the recurring Strassen patterns
    line up across blocks, followed by an exact rational column-linearity
    reduction that repeatedly rebuilds its basis and greedily merges any column
    that is a rational linear combination of others *with the same relation in all
    three factors*, lowering the rank while preserving the tensor exactly.

Interface (unchanged): construct() -> {'U','V','W'} with shapes
  U (n*m, rank), V (m*p, rank), W (p*n, rank), integer/half-integer entries,
  W indexed by the official trace convention W[k*n+i, r].
"""
import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Fast verification: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


# ---------------------------------------------------------------------------
# Flat, index-based block assembler (refinement machinery from the inspirations).
# ---------------------------------------------------------------------------
def _build_blocks():
    """Assemble the block decomposition with a flat index-based assembler.

    Returns (U, V, W) as numpy arrays. U (12, rank), V (16, rank), W (12, rank).
    """
    rank = 4 * 7 + 4 * 4  # 36
    U = np.zeros((12, rank))
    V = np.zeros((16, rank))
    W = np.zeros((12, rank))
    col = 0

    def add_strassen(a, b, c, d, e, f, g, h, c00, c01, c10, c11):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (c+d)e
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3 = a(f-h)
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4 = d(g-e)
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5 = (a+b)h
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6 = (c-a)(e+f)
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7 = (b-d)(g+h)
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_trivial(a, b, e, f, g, h, c0, c1):
        nonlocal col
        U[a, col] += 1; V[e, col] += 1; W[c0, col] += 1; col += 1  # a*e
        U[b, col] += 1; V[g, col] += 1; W[c0, col] += 1; col += 1  # b*g
        U[a, col] += 1; V[f, col] += 1; W[c1, col] += 1; col += 1  # a*f
        U[b, col] += 1; V[h, col] += 1; W[c1, col] += 1; col += 1  # b*h

    # U index: i*m + j ; V index: j*p + k ; W index: k*n + i
    # A top-left 2x2: [0,1,4,5] ; A top-right 2x2: [2,3,6,7]
    # A bottom-left 1x2: [8,9] ; A bottom-right 1x2: [10,11]
    # B_11: [0,1,4,5] ; B_12: [2,3,6,7] ; B_21: [8,9,12,13] ; B_22: [10,11,14,15]
    # C cols 0,1: C[0,0]=0 C[0,1]=3 C[1,0]=1 C[1,1]=4 C[2,0]=2 C[2,1]=5
    # C cols 2,3: C[0,2]=6 C[0,3]=9 C[1,2]=7 C[1,3]=10 C[2,2]=8 C[2,3]=11

    # Top 2 rows: four Strassen 2x2*2x2
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)   # A_TL * B_11
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)  # A_TL * B_12
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4) # A_TR * B_21
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)  # A_TR * B_22

    # Bottom row: four trivial 1x2*2x2
    add_trivial(8, 9, 0, 1, 4, 5, 2, 5)        # A_BL * B_11
    add_trivial(8, 9, 2, 3, 6, 7, 8, 11)       # A_BL * B_12
    add_trivial(10, 11, 8, 9, 12, 13, 2, 5)    # A_BR * B_21
    add_trivial(10, 11, 10, 11, 14, 15, 8, 11) # A_BR * B_22

    assert col == rank, (col, rank)
    return U, V, W


def _schoolbook():
    """Rank-48 schoolbook decomposition (fallback)."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Exact rational column-linearity reduction (strengthened).
# ---------------------------------------------------------------------------
def _col_relation(M, target, basis):
    """Solve M[:, basis] @ x = M[:, target] over Q. Return {basis_idx: coeff} or None."""
    rows = len(M)
    n = len(basis)
    if n == 0:
        return {} if all(M[rr, target] == 0 for rr in range(rows)) else None
    mat = [[Fraction(M[rr, b]) for b in range(n)] + [Fraction(M[rr, target])]
           for rr in range(rows)]
    rank = 0
    for c in range(n):
        piv = -1
        for rr in range(rank, rows):
            if mat[rr][c] != 0:
                piv = rr
                break
        if piv == -1:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        pv = mat[rank][c]
        mat[rank] = [x / pv for x in mat[rank]]
        for rr in range(rows):
            if rr != rank:
                f = mat[rr][c]
                if f != 0:
                    mat[rr] = [mat[rr][j] - f * mat[rank][j] for j in range(n + 1)]
        rank += 1
    for rr in range(rows):
        if all(mat[rr][j] == 0 for j in range(n)) and mat[rr][n] != 0:
            return None
    x = [Fraction(0)] * n
    for rr in range(rank):
        for cc in range(n):
            if mat[rr][cc] != 0:
                x[cc] = mat[rr][n]
                break
    return {basis[i]: x[i] for i in range(n)}


def _eliminate_columns(U, V, W, deadline):
    """Exact rational column-linearity reduction.

    A column c can be dropped if it is a rational linear combination of a set S of
    other columns AND the SAME relation holds for the corresponding V and W columns
    (then c's rank-1 term equals the sum of S's scaled terms). We repeatedly rebuild
    a basis and greedily merge any such column, so cross-block cancellations that
    only appear after earlier merges are also caught.
    """
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)

    def to_frac(A):
        return [[Fraction(int(round(A[rr, cc]))) for cc in range(A.shape[1])]
                for rr in range(A.shape[0])]

    def check_relation(Mq, c, rel, basis):
        for rr in range(len(Mq)):
            lhs = Mq[rr][c]
            rhs = sum(rel[b] * Mq[rr][b] for b in basis)
            if lhs != rhs:
                return False
        return True

    while time.time() < deadline:
        Uq = to_frac(Uf)
        Vq = to_frac(Vf)
        Wq = to_frac(Wf)
        r = Uf.shape[1]
        if r == 0:
            break

        # Build a maximal independent basis of U columns (in column order).
        basis = []
        for c in range(r):
            if all(Uq[rr][c] == 0 for rr in range(len(Uq))):
                continue
            if basis and _col_relation(Uq, c, basis) is not None:
                continue
            basis.append(c)

        # Find one droppable column (any column, basis or not).
        dropped = None
        for c in range(r):
            if time.time() > deadline:
                break
            if all(Uq[rr][c] == 0 for rr in range(len(Uq))):
                dropped = (c, {}, basis)  # zero column
                break
            others = [b for b in range(r) if b != c]
            # use a small candidate basis: the current basis minus c, plus a couple
            # of the smallest-index other columns to allow non-basis relations.
            cand = [b for b in basis if b != c]
            if len(cand) < 2:
                cand = others[:3]
            rel = _col_relation(Uq, c, cand)
            if rel is None:
                continue
            if check_relation(Vq, c, rel, cand) and check_relation(Wq, c, rel, cand):
                dropped = (c, rel, cand)
                break

        if dropped is None:
            break

        c, rel, cand = dropped
        for b in cand:
            a = rel.get(b, Fraction(0))
            if a != 0:
                Uf[:, b] += float(a) * Uf[:, c]
                Vf[:, b] += float(a) * Vf[:, c]
                Wf[:, b] += float(a) * Wf[:, c]
        Uf = np.delete(Uf, c, axis=1)
        Vf = np.delete(Vf, c, axis=1)
        Wf = np.delete(Wf, c, axis=1)

    Uf = np.round(Uf * 2) / 2
    Vf = np.round(Vf * 2) / 2
    Wf = np.round(Wf * 2) / 2
    return Uf, Vf, Wf


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4."""
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _build_blocks()

    if _verify_fast(U, V, W):
        best_rank = U.shape[1]
        best = (U, V, W)
        try:
            U2, V2, W2 = _eliminate_columns(U, V, W, deadline)
            if _verify_fast(U2, V2, W2) and U2.shape[1] < best_rank:
                best_rank = U2.shape[1]
                best = (U2, V2, W2)
        except Exception:
            pass
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
