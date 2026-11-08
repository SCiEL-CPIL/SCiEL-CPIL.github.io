"""Exact rank-reduction search for the 2x3x5 matrix-multiplication tensor.

Strategy (different from the seed's fixed block-partition construction):
  1. Build a known valid decomposition (Strassen 2x2x2 sub-block + schoolbook
     leftovers) at rank 29, verified exactly over Q.
  2. Iteratively attempt to drop one rank term at a time: for a candidate
     column to remove, set its entries to 0 and solve the resulting linear
     system over the rationals for the free (still-unknown) entries so that
     every tensor entry still equals 1. Accept the reduction only if a
     consistent exact solution exists.
  3. Repeat until no single-term removal works or the time budget is hit.

The linear system is tiny (<=30 equations in <=12 unknowns), so exact
fraction Gaussian elimination is fast and safe within the 150 s budget.
"""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]


def _half(M):
    out = []
    for row in M:
        r = []
        for v in row:
            if isinstance(v, Fraction):
                f = v * 2
                if f.denominator == 1:
                    fv = f.numerator
                    r.append(fv // 2 if fv % 2 == 0 else fv / 2)
                else:
                    r.append(float(v))
            else:
                r.append(v)
        out.append(r)
    return out


def _check(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(rank):
                    a = uc[c]
                    if a:
                        s += a * vc[c] * wc[c]
                if s != 1:
                    return False
    return True


def _build_rank29(n, m, p):
    """Strassen 2x2x2 on (i in {0,1}, j in {0,1}, k in {0,1}) + schoolbook rest."""
    bini = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        (0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0),
        (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0),
    ]
    rank = 29
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for (u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11) in bini:
        for i in range(2):
            U[i * m + 0][col] = Fraction((u00, u01, u10, u11)[i * 2 + 0])
            U[i * m + 1][col] = Fraction((u00, u01, u10, u11)[i * 2 + 1])
        for j in range(2):
            V[j * p + 0][col] = Fraction((v00, v01, v10, v11)[j * 2 + 0])
            V[j * p + 1][col] = Fraction((v00, v01, v10, v11)[j * 2 + 1])
        for k in range(2):
            W[k * n + 0][col] = Fraction((w00, w01, w10, w11)[k * 2 + 0])
            W[k * n + 1][col] = Fraction((w00, w01, w10, w11)[k * 2 + 1])
        col += 1
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    assert col == rank, (col, rank)
    return U, V, W


def _gauss(A, b, n):
    """Solve A x = b over Q. A is n x n list-of-lists of Fraction, b is list.
    Returns solution list or None if inconsistent."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv == -1:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def _try_remove(U, V, W, n, m, p, c):
    """Attempt to set rank column c to zero and solve for the free entries.
    Returns (newU, newV, newW) on success, else None."""
    # Free entries: those (row, c) currently zero that are not forced to stay zero.
    # We allow any currently-zero entry in column c to become a variable.
    free = []  # list of (matrix, row, idx_in_free)
    for mat, rows in ((U, n * m), (V, m * p), (W, p * n)):
        for r in range(rows):
            if mat[r][c] == 0:
                free.append((mat, r, len(free)))
    nfree = len(free)

    # Build linear system: for each (i,j,k),
    #   sum_{c' != c} U[i,j,c'] V[j,k,c'] W[k,i,c'] + sum_{free} coeff * x = 1
    # => sum_free coeff * x = 1 - sum_{c' != c} ...
    eqs = []
    rhs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                s = Fraction(0)
                for c2 in range(len(uc)):
                    if c2 == c:
                        continue
                    s += uc[c2] * vc[c2] * wc[c2]
                # coefficients for free variables in column c
                row_coeff = []
                for (mat, r, idx) in free:
                    if mat is U:
                        coef = vc[c] * wc[c]
                    elif mat is V:
                        coef = uc[c] * wc[c]
                    else:
                        coef = uc[c] * vc[c]
                    row_coeff.append(coef)
                eqs.append(row_coeff)
                rhs.append(1 - s)

    # We have n*m*p equations and nfree unknowns. Need a consistent solution.
    # If nfree == 0: only valid if all rhs == 0.
    if nfree == 0:
        if all(r == 0 for r in rhs):
            # column c is already zero everywhere; removing it is trivial
            newU = [row[:c] + row[c + 1:] for row in U]
            newV = [row[:c] + row[c + 1:] for row in V]
            newW = [row[:c] + row[c + 1:] for row in W]
            return newU, newV, newW
        return None

    # Pick a square subsystem: try to find nfree independent equations.
    # Simple approach: take the first nfree equations and check if the
    # resulting square system is consistent with ALL equations.
    # To be robust, do a full consistency check via row-reduction on the
    # augmented system (nfree unknowns, n*m*p equations).
    # Build augmented rows: [coeffs... | rhs]
    aug = []
    for rowc, r in zip(eqs, rhs):
        aug.append(rowc + [r])
    # Row-reduce to check consistency.
    ncols = nfree + 1
    rank_r = 0
    rows = [row[:] for row in aug]
    for col in range(nfree):
        piv = -1
        for r in range(rank_r, len(rows)):
            if rows[r][col] != 0:
                piv = r
                break
        if piv == -1:
            continue
        rows[rank_r], rows[piv] = rows[piv], rows[rank_r]
        pv = rows[rank_r][col]
        for j in range(col, ncols):
            rows[rank_r][j] /= pv
        for r in range(len(rows)):
            if r != rank_r and rows[r][col] != 0:
                f = rows[r][col]
                for j in range(col, ncols):
                    rows[r][j] -= f * rows[rank_r][j]
        rank_r += 1
    # Consistency: any row with all-zero coeffs but nonzero rhs => inconsistent
    for r in range(rank_r, len(rows)):
        if rows[r][nfree] != 0:
            return None
    # Extract solution from the reduced system (free vars set to 0)
    sol = [Fraction(0)] * nfree
    for r in range(rank_r):
        # find pivot column
        pc = -1
        for col in range(nfree):
            if rows[r][col] != 0:
                pc = col
                break
        if pc != -1:
            sol[pc] = rows[r][nfree]

    # Apply solution
    for (mat, r, idx) in free:
        mat[r][c] = sol[idx]
    # Set column c of the fixed (non-free) entries to zero
    for mat in (U, V, W):
        for r in range(len(mat)):
            if (mat, r, 0) not in free:
                mat[r][c] = Fraction(0)
    # Actually simpler: zero the whole column c, then the free ones were already set.
    for mat in (U, V, W):
        for r in range(len(mat)):
            mat[r][c] = Fraction(0)
    for (mat, r, idx) in free:
        mat[r][c] = sol[idx]

    # Verify
    if _check(U, V, W, n, m, p):
        newU = [row[:c] + row[c + 1:] for row in U]
        newV = [row[:c] + row[c + 1:] for row in V]
        newW = [row[:c] + row[c + 1:] for row in W]
        return newU, newV, newW
    return None


def _deepcopy(U, V, W):
    return (
        [[Fraction(x) for x in row] for row in U],
        [[Fraction(x) for x in row] for row in V],
        [[Fraction(x) for x in row] for row in W],
    )


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin under 150 s

    U, V, W = _build_rank29(n, m, p)
    if not _check(U, V, W, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": _half(U), "V": _half(V), "W": _half(W)}

    # Iteratively try to remove one term at a time.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rank = len(U[0])
        for c in range(rank - 1, -1, -1):  # try from the end (schoolbook cols first)
            if time.time() > deadline:
                break
            U0, V0, W0 = _deepcopy(U, V, W)
            res = _try_remove(U0, V0, W0, n, m, p, c)
            if res is not None:
                U, V, W = res
                improved = True
                break  # restart scan after a reduction

    # Final safety: verify; if broken, fall back to the rank-29 build.
    if not _check(U, V, W, n, m, p):
        U, V, W = _build_rank29(n, m, p)

    return {"U": _half(U), "V": _half(V), "W": _half(W)}
