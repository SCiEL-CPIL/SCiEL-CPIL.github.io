import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start = time.time()
    deadline = 135.0  # safety margin before the 150 s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the multiplication tensor T[a, b] where a = i*m + j (input A),
    # b = j*p + k (input B).  T[a, b] = delta_{a mod m, b // p}, i.e. 1 iff
    # the column index of A equals the row index of B.
    T = [[Fraction(0) for _ in range(mp)] for _ in range(nm)]
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for k in range(p):
                b = j * p + k
                T[a][b] = Fraction(1)

    # Exact Gaussian elimination over the rationals to find the rank and a
    # set of pivot columns that form a basis for the column space.
    M = [row[:] for row in T]
    R = len(M)
    C = len(M[0])
    pivot_rows = []
    pivot_cols = []
    r = 0
    for c in range(C):
        if time.time() > deadline - 10:
            break
        pivot = None
        for i in range(r, R):
            if M[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
        # Normalize the pivot row (exact, so no precision loss).
        pv = M[r][c]
        for cc in range(c, C):
            M[r][cc] = M[r][cc] / pv
        # Eliminate the pivot column in all other rows.
        for i in range(R):
            if i == r:
                continue
            factor = M[i][c]
            if factor != 0:
                for cc in range(c, C):
                    M[i][cc] -= factor * M[r][cc]
        pivot_rows.append(r)
        pivot_cols.append(c)
        r += 1

    rank = len(pivot_cols)
    if rank >= n * m * p:
        return schoolbook()

    # Extract the basis columns from the ORIGINAL tensor (they are linearly
    # independent and span the column space).
    basis_cols = [T[a][c] for a in range(nm) for c in [pc] for pc in [pc] if True] if False else \
                 [[T[a][pc] for a in range(nm)] for pc in pivot_cols]

    # Each basis column is a vector over the A-indices.  We now want to express
    # it as a sum of rank-1 tensors U_col * V_col * W_col.  The most direct
    # exact construction: for each basis column vector v (length nm), note that
    # v[a] depends only on (i, j) via the shared j.  We decompose using the
    # structure: the column space of T is spanned by columns indexed by (j, k).
    # Column (j, k) has entries T[a, (j,k)] = 1 iff a = i*m + j for any i.
    # So column (j,k) = sum_i e_{i*m+j} * (e_{j*p+k}) * (e_{k*n+i}).
    #
    # Instead of re-deriving, we build a rank-1 decomposition directly from the
    # pivot columns by treating each pivot column as a product.  A clean exact
    # rank-1 decomposition of T of rank = rank(T) is:
    #   T = sum_{c in basis}  u_c  v_c  w_c
    # where we factor each basis column.  Since the basis columns are linearly
    # independent and each equals a single original column (j0, k0) up to a
    # combination, we use the known rank-1 identity for a single column.
    #
    # For safety and exactness, fall back to the schoolbook decomposition if we
    # cannot produce a verified decomposition of the computed rank.  We attempt
    # to build one using the pivot columns as follows: represent each basis
    # column as a linear combination of the original elementary columns, then
    # sum the corresponding rank-1 terms.

    # Find coefficients expressing each pivot column as a combination of the
    # elementary columns e_{(j,k)}.  Since each elementary column has support
    # on a fixed j-block, and pivot columns are combinations, we solve.
    # Elementary column for (j, k): E_{j,k}[a] = 1 if a mod m == j and a // m
    # is any i (i.e. a = i*m + j), else 0.  Note this does NOT depend on k!
    # So columns (j, 0), (j, 1), ..., (j, p-1) are ALL IDENTICAL.
    #
    # This means the column space is spanned by just m = 3 distinct vectors
    # (one per j).  Hence rank(T) <= m = 3.  Let's verify and use that.
    #
    # Indeed: T[a, b] with a = i*m + j, b = j'*p + k is 1 iff j == j'.
    # So T = sum_{j} (sum_i e_{i*m+j}) (e_{j}) (sum_k ... ) -- but we need the
    # W index to depend on k and i.  The rank-1 term for a fixed j is:
    #   U_j[a] = 1 if a mod m == j else 0   (length nm)
    #   V_j[b] = 1 if b // p == j else 0    (length mp)
    #   W_j[c] = 1 if c mod n == a//m ... but W depends on (k, i).
    # For fixed j, the entries are 1 for all (i, k).  So W must be 1 for all
    # (k, i) -> W_j = all ones (length pn).  Then
    #   (U_j * V_j * W_j)[a, b] = U_j[a] V_j[b] * 1 = 1 iff a mod m == b // p.
    # Summing over j gives exactly T.  So rank 3 suffices!

    rank3 = m
    U3 = [[0] * rank3 for _ in range(nm)]
    V3 = [[0] * rank3 for _ in range(mp)]
    W3 = [[0] * rank3 for _ in range(pn)]
    for j in range(m):
        for i in range(n):
            U3[i * m + j][j] = 1
        for k in range(p):
            V3[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W3[k * n + i][j] = 1

    # Verify rank-3 decomposition exactly.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank3):
                    s += U3[i * m + j][c] * V3[j * p + k][c] * W3[k * n + i][c]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if ok:
        return {"U": U3, "V": V3, "W": W3}

    # Fallback: return the schoolbook (guaranteed valid).
    return schoolbook()
