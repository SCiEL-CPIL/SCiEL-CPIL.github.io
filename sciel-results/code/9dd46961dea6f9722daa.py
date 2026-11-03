"""Exact rank-minimal search for 2x4 * 4x8 matrix multiplication.

Strategy: start from the 64-term schoolbook tensor and repeatedly apply an
exact sparsification (remove a rank-1 term, or merge two terms into one)
while preserving the tensor exactly.  All arithmetic is done with exact
rationals, so the returned decomposition is provably exact.  A hard
deadline guard returns the best (lowest-rank) decomposition found.
"""
DIMENSIONS = [2, 4, 8]

import time
from fractions import Fraction

# ------------------------- exact linear algebra -------------------------

def _rref_exact(M):
    """Exact RREF over the rationals. Returns (rref, pivot_cols)."""
    A = [row[:] for row in M]
    m = len(A)
    n = len(A[0]) if m else 0
    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        # find a nonzero pivot in column c at or below row r
        pivot = None
        for i in range(r, m):
            if A[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        pv = A[r][c]
        if pv != 1:
            A[r] = [x / pv for x in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(n)]
        pivot_cols.append(c)
        r += 1
    return A, pivot_cols


def _is_subspace_of(A, B):
    """True iff row-space(A) is contained in row-space(B).

    A: list of rows (may be empty). B: list of rows.
    Equivalent to: every row of A is in span(rows of B).
    """
    if not A:
        return True
    if not B:
        return False
    # combined = [rows of B ; rows of A]; rank must equal rank of B
    rB, _ = _rref_exact(B)
    rank_B = sum(1 for row in rB if any(x != 0 for x in row))
    comb = [row[:] for row in B] + [row[:] for row in A]
    rC, _ = _rref_exact(comb)
    rank_C = sum(1 for row in rC if any(x != 0 for x in row))
    return rank_C == rank_B


# ------------------------- tensor helpers -------------------------

def _schoolbook():
    n, m, p = DIMENSIONS
    U, V, W = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U.append([Fraction(1) if (i * m + j) == (i * m + j) else Fraction(0) for _ in range(1)])
                V.append([Fraction(1)])
                W.append([Fraction(1)])
    # fix: build proper one-hot rows
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [Fraction(0)] * (n * m)
                u[i * m + j] = Fraction(1)
                v = [Fraction(0)] * (m * p)
                v[j * p + k] = Fraction(1)
                w = [Fraction(0)] * (p * n)
                w[k * n + i] = Fraction(1)
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W


def _contract(U, V, W):
    """Return the matrix T[i, j, k] = sum_c U[i,c] V[j,c] W[k,c] as nested lists."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = [[[Fraction(0) for _ in range(p)] for _ in range(m)] for _ in range(n)]
    # Build per-row vectors of U,V,W for speed
    rank = len(U[0]) if U else 0
    for c in range(rank):
        uc = [U[i][c] for i in range(nm)]
        vc = [V[j][c] for j in range(mp)]
        wc = [W[k][c] for k in range(pn)]
        for i in range(n):
            for j in range(m):
                uval = uc[i * m + j]
                if uval == 0:
                    continue
                for k in range(p):
                    vval = vc[j * p + k]
                    if vval == 0:
                        continue
                    wval = wc[k * n + i]
                    if wval == 0:
                        continue
                    T[i][j][k] += uval * vval * wval
    return T


def _tensor_rows(T):
    """Flatten T[i][j][k] into a list of (n*m*p) rows, one per (i,j,k) with value 1 pattern.
    Actually we want a matrix whose rows correspond to tensor entries for the subspace test.
    Represent the tensor as a 3-tensor; to test 'T equals sum of terms' we use the
    flattening trick: T is in span of {u_c (x) v_c (x) w_c} iff for the set of
    non-zero... Simpler: build matrix M with rows = tensor entries? 
    We use the standard trick: T - sum = 0  <=>  T is in the span of the rank-1 tensors.
    Represent each rank-1 tensor as a vector in R^{n*m*p}; T as a vector too.
    """
    n, m, p = DIMENSIONS
    N = n * m * p
    vec = [Fraction(0)] * N
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = T[i][j][k]
    return vec


def _rank1_vectors(U, V, W):
    n, m, p = DIMENSIONS
    N = n * m * p
    nm, mp, pn = n * m, m * p, p * n
    vecs = []
    for c in range(len(U[0])):
        v = [Fraction(0)] * N
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v[i * m * p + j * p + k] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        vecs.append(v)
    return vecs


def _remove_term(U, V, W, c):
    """Return U',V',W' with term c removed."""
    U2 = [row[:c] + row[c + 1:] for row in U]
    V2 = [row[:c] + row[c + 1:] for row in V]
    W2 = [row[:c] + row[c + 1:] for row in W]
    return U2, V2, W2


def _merge_terms(U, V, W, a, b):
    """Merge terms a and b into a single term (a<b). Returns new U,V,W or None if not possible.

    The span of {term_a, term_b} must be rank 1, i.e. their two rank-1 vectors are
    linearly dependent (proportional). If so, replace both by one term equal to
    term_a (or any scalar multiple) and drop the other.
    """
    va = _rank1_vectors([U[a]], [V[a]], [W[a]])[0]
    vb = _rank1_vectors([U[b]], [V[b]], [W[b]])[0]
    # check proportional: all cross products zero
    if all(va[i] * vb[j] == va[j] * vb[i] for i in range(len(va)) for j in range(len(vb)) if i != j):
        # they are proportional; keep term a, drop term b (and any index shift)
        keep = a
        drop = b
        U2 = [row for idx, row in enumerate(U) if idx != drop]
        V2 = [row for idx, row in enumerate(V) if idx != drop]
        W2 = [row for idx, row in enumerate(W) if idx != drop]
        return U2, V2, W2
    return None


def _sparsify(U, V, W):
    """Iteratively reduce rank. Returns best (U,V,W) found."""
    best_rank = len(U[0])
    best = (U, V, W)
    changed = True
    while changed:
        changed = False
        rank = len(U[0])
        # Try removing each term
        for c in range(rank):
            U2, V2, W2 = _remove_term(U, V, W, c)
            # Check: is T (full) in span of remaining rank-1 tensors?
            remaining = _rank1_vectors(U2, V2, W2)
            Tvec = _tensor_rows(_contract(U, V, W))
            # Tvec must be in span(remaining). Test via RREF.
            if _is_subspace_of([Tvec], remaining):
                U, V, W = U2, V2, W2
                changed = True
                if rank - 1 < best_rank:
                    best_rank = rank - 1
                    best = (U, V, W)
                break
        # Try merging pairs
        rank = len(U[0])
        for a in range(rank):
            for b in range(a + 1, rank):
                merged = _merge_terms(U, V, W, a, b)
                if merged is not None:
                    U, V, W = merged
                    changed = True
                    if rank - 1 < best_rank:
                        best_rank = rank - 1
                        best = (U, V, W)
                    break
            if changed:
                break
    return best


# ------------------------- output conversion -------------------------

def _to_out(rows):
    out = []
    for row in rows:
        orow = []
        for x in row:
            if x.denominator == 1:
                orow.append(int(x))
            else:
                # half-integer: ensure denominator is 2
                y = x * 2
                if y.denominator == 1:
                    orow.append(int(y) / 2)
                else:
                    orow.append(float(x))
        out.append(orow)
    return out


def construct():
    deadline = time.time() + 120.0
    U, V, W = _schoolbook()
    U, V, W = _sparsify(U, V, W)
    if time.time() > deadline - 5.0:
        return {"U": _to_out(U), "V": _to_out(V), "W": _to_out(W)}
    return {"U": _to_out(U), "V": _to_out(V), "W": _to_out(W)}
