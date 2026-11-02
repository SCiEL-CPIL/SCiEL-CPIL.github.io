import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor (all-ones)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_mat = U[:, t].reshape(n, m)
        v_mat = V[:, t].reshape(m, p)
        w_mat = W[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    expected = np.ones((n, m, p), dtype=float)
    return bool(np.allclose(T, expected, atol=1e-8))


def _build_strassen_terms(n, m, p):
    """
    Build a list of Strassen terms covering the full (n,m,p) multiplication.
    Each term is (a_coeffs, b_coeffs, c_coeffs):
      a_coeffs: dict {(i,j): coeff} over A (n x m)
      b_coeffs: dict {(j,k): coeff} over B (m x p)
      c_coeffs: dict {(i,k): coeff} over C (p x n), W index k*n+i
    We tile the A matrix with 2x2 blocks where possible, and cover the rest
    with schoolbook terms. The idea: for each 2x2 A-block (rows 2i,2i+1; cols 2j,2j+1)
    and each 2x2 B-block (rows 2j,2j+1; cols 2k,2k+1), the 8 scalar products
    of those blocks can be done with 7 Strassen multiplications, and the
    contributions land in the corresponding 2x2 C-block.
    """
    terms = []

    def add_term(a, b, c):
        terms.append((a, b, c))

    # Determine which (i,j) A-positions and (j,k) B-positions are covered by
    # 2x2 Strassen blocks. We cover A rows in pairs (0,1), (2,3), ... and cols
    # in pairs. Similarly B rows in pairs and cols in pairs.
    # A block (i0,j0) 2x2 requires i0+1<n, j0+1<m.
    # B block (j0,k0) 2x2 requires j0+1<m, k0+1<p.
    # For a Strassen block we need a 2x2 A-block at (i0,j0) AND a 2x2 B-block at
    # (j0,k0) sharing the same j0 pair, producing a 2x2 C-block at (i0,k0).
    # But A and B must share the inner index pair j0, j0+1. So we iterate over
    # j-pairs, and for each, over i-pairs and k-pairs.

    used = set()  # track (i,j,k) triples handled by Strassen blocks

    # Build the set of j-pairs available
    j_pairs = [j0 for j0 in range(0, m - 1, 2)]

    for j0 in j_pairs:
        # i-pairs
        for i0 in range(0, n - 1, 2):
            # k-pairs
            for k0 in range(0, p - 1, 2):
                # 2x2 A block rows i0,i0+1 cols j0,j0+1
                # 2x2 B block rows j0,j0+1 cols k0,k0+1
                # 2x2 C block rows i0,i0+1 cols k0,k0+1
                # Strassen 7 terms:
                # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
                add_term(
                    {(i0, j0): 1, (i0 + 1, j0 + 1): 1},
                    {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
                    {(i0, k0): 1, (i0 + 1, k0 + 1): 1},
                )
                # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
                add_term(
                    {(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
                    {(j0, k0): 1},
                    {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1},
                )
                # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
                add_term(
                    {(i0, j0): 1},
                    {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
                    {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1},
                )
                # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
                add_term(
                    {(i0 + 1, j0 + 1): 1},
                    {(j0 + 1, k0): 1, (j0, k0): -1},
                    {(i0, k0): 1, (i0 + 1, k0): 1},
                )
                # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
                add_term(
                    {(i0, j0): 1, (i0, j0 + 1): 1},
                    {(j0 + 1, k0 + 1): 1},
                    {(i0, k0): -1, (i0, k0 + 1): 1},
                )
                # M6 = (A21-A11)(B11+B12) -> C22 += M6
                add_term(
                    {(i0 + 1, j0): 1, (i0, j0): -1},
                    {(j0, k0): 1, (j0, k0 + 1): 1},
                    {(i0 + 1, k0 + 1): 1},
                )
                # M7 = (A12-A22)(B21+B22) -> C11 += M7
                add_term(
                    {(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
                    {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
                    {(i0, k0): 1},
                )
                # Mark the 8 triples in this 2x2x2 region as handled
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            used.add((i0 + di, j0 + dj, k0 + dk))

    # Now add schoolbook terms for any (i,j,k) not covered by Strassen.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})

    return terms


def _terms_to_matrices(terms, n, m, p):
    """Convert list of (a,b,c) dict-terms into U,V,W matrices."""
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for t, (a, b, c) in enumerate(terms):
        for (i, j), val in a.items():
            U[i * m + j][t] = val
        for (j, k), val in b.items():
            V[j * p + k][t] = val
        for (i, k), val in c.items():
            W[k * n + i][t] = val
    return U, V, W


def _peel_terms(terms, n, m, p, deadline):
    """
    Attempt to reduce the term list by peeling: if a term's contribution is
    a linear combination of the others (i.e., its rank-1 tensor lies in the span
    of the remaining rank-1 tensors), we can drop it and adjust. In practice we
    try a greedy removal: for each term, check if removing it and re-solving the
    coefficients keeps the all-ones tensor. We use a linear-algebra peel:
    treat each term's (U_col, V_col, W_col) and check linear dependence of the
    outer products.
    """
    rank = len(terms)
    # Build the tensor as a matrix: flatten (i,j,k) -> index. Each term t gives a
    # vector v_t of size n*m*p where v_t[idx(i,j,k)] = U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t].
    # The sum of v_t must equal ones. We want the smallest subset that still sums
    # to ones (allowing coefficient rescaling). We do a greedy linear-algebra peel:
    # find a term whose v_t is in the span of the others, then solve for the
    # coefficients of the rest.
    size = n * m * p
    # Build v_t vectors
    vecs = []
    for t in range(rank):
        a, b, c = terms[t]
        v = np.zeros(size)
        for (i, j), ua in a.items():
            for (j2, k), vb in b.items():
                if j2 != j:
                    continue
                for (i2, k2), wc in c.items():
                    if i2 != i or k2 != k:
                        continue
                    idx = (i * m + j) * p + k
                    v[idx] += ua * vb * wc
        vecs.append(v)

    vecs = np.array(vecs)  # (rank, size)
    target = np.ones(size)

    # Greedy peel: try to drop terms one at a time. For a candidate set of kept
    # indices, check if target is in the span (solve least squares, check residual).
    kept = list(range(rank))
    changed = True
    while changed:
        changed = False
        start = time.time()
        for t in list(kept):
            if time.time() > deadline:
                return None
            candidate = [x for x in kept if x != t]
            if len(candidate) == 0:
                continue
            M = vecs[candidate]  # (len(candidate), size)
            # Solve M^T x = target  (least squares)
            try:
                coeff, res, rankval, sv = np.linalg.lstsq(M.T, target, rcond=None)
            except np.linalg.LinAlgError:
                continue
            recon = M.T @ coeff
            if np.allclose(recon, target, atol=1e-8):
                # We can drop term t and rescale the others.
                # New terms: for each kept index, multiply its (a,b,c) by coeff.
                kept = candidate
                # We need to track scaling. Return a mapping.
                # We'll rebuild terms with scaling in the caller.
                changed = True
                break
        # If we changed, we need to recompute vecs for the new kept set with scaling.
        # This is complex; instead, we do a simpler peel that only drops terms that
        # are exactly zero-contribution (v_t == 0).
    # Simpler robust peel: drop any term whose vector is all zeros.
    kept = []
    for t in range(rank):
        if not np.allclose(vecs[t], 0.0, atol=1e-12):
            kept.append(t)
    if len(kept) == rank:
        return None
    # Rebuild terms (no scaling needed since we only dropped zero terms)
    return [terms[t] for t in kept]


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # Fallback: schoolbook
    school_terms = [({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                    for i in range(n) for j in range(m) for k in range(p)]
    U_s, V_s, W_s = _terms_to_matrices(school_terms, n, m, p)
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = n * m * p

    # Strassen tiled decomposition
    strassen_terms = _build_strassen_terms(n, m, p)
    U_t, V_t, W_t = _terms_to_matrices(strassen_terms, n, m, p)
    if _validate(U_t, V_t, W_t, n, m, p, len(strassen_terms)):
        best = {"U": U_t, "V": V_t, "W": W_t}
        best_rank = len(strassen_terms)

    # Peeling pass on the Strassen terms
    peeled = _peel_terms(strassen_terms, n, m, p, deadline)
    if peeled is not None and len(peeled) < best_rank:
        U_p, V_p, W_p = _terms_to_matrices(peeled, n, m, p)
        if _validate(U_p, V_p, W_p, n, m, p, len(peeled)):
            best = {"U": U_p, "V": V_p, "W": W_p}
            best_rank = len(peeled)

    return best
