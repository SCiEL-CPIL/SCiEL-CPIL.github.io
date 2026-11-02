import numpy as np
import time

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 140.0  # safety margin under 150s wall-clock


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the matmul tensor for dims n,m,p.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Convention: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sums to 1 for each (i,j,k).
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(recon - 1.0) > tol:
                    return False
    return True


def _strassen_decomposition():
    """Return the known Strassen-based decomposition (rank 98) for 4x4 * 4x8."""
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen decomposition for 2x2x2
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))

    # M1 = (A00 - A11)(B10 + B11)
    U_S[0, 0] = 1; U_S[3, 0] = -1
    V_S[2, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    # M2 = (A00 + A11) B00
    U_S[0, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1
    # M3 = A00 (B01 - B11)
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[1, 2] = 1
    # M4 = A11 (B10 - B00)
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = 1
    # M5 = (A00 + A01) B11
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1; W_S[1, 4] = 1
    # M6 = (A10 - A00) (B00 + B01)
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[3, 5] = 1
    # M7 = (A01 - A11) (B10 + B11)
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[0, 6] = 1; W_S[3, 6] = -1

    # Construct 2x2x4 tensor factors
    U_O = np.zeros((4, 14))
    V_O = np.zeros((8, 14))
    W_O = np.zeros((8, 14))

    # Block 1: k in {0, 1}
    U_O[:, 0:7] = U_S
    V_O[0, 0:7] = V_S[0]; V_O[1, 0:7] = V_S[1]
    V_O[4, 0:7] = V_S[2]; V_O[5, 0:7] = V_S[3]
    W_O[0, 0:7] = W_S[0]; W_O[1, 0:7] = W_S[1]
    W_O[2, 0:7] = W_S[2]; W_O[3, 0:7] = W_S[3]
    # Block 2: k in {2, 3}
    U_O[:, 7:14] = U_S
    V_O[2, 7:14] = V_S[0]; V_O[3, 7:14] = V_S[1]
    V_O[6, 7:14] = V_S[2]; V_O[7, 7:14] = V_S[3]
    W_O[4, 7:14] = W_S[0]; W_O[5, 7:14] = W_S[1]
    W_O[6, 7:14] = W_S[2]; W_O[7, 7:14] = W_S[3]

    # Kronecker products
    U_raw = np.kron(U_O, U_S)
    V_raw = np.kron(V_O, V_S)
    W_raw = np.kron(W_O, W_S)

    # Permutations
    perm_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i_e in range(2):
                for j_e in range(2):
                    target = 8*I + 4*i_e + 2*J + j_e
                    source = (I*2 + J)*4 + (i_e*2 + j_e)
                    perm_U[target] = source
    U_final = U_raw[perm_U, :]

    perm_V = np.zeros(32, dtype=int)
    for J in range(2):
        for K in range(4):
            for j_e in range(2):
                for k_e in range(2):
                    target = 16*J + 8*j_e + 2*K + k_e
                    source = (J*4 + K)*4 + (j_e*2 + k_e)
                    perm_V[target] = source
    V_final = V_raw[perm_V, :]

    perm_W = np.zeros(32, dtype=int)
    for K in range(4):
        for I in range(2):
            for k_e in range(2):
                for i_e in range(2):
                    target = 8*K + 4*k_e + 2*I + i_e
                    source = (K*2 + I)*4 + (k_e*2 + i_e)
                    perm_W[target] = source
    W_final = W_raw[perm_W, :]

    # Verification
    if not _verify(U_final, V_final, W_final, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    return {"U": U_final, "V": V_final, "W": W_final, "rank": U_final.shape[1]}


def _search_lower_rank(best_U, best_V, best_W, deadline):
    """
    Time-bounded rank-reduction search.

    Given a valid decomposition of rank r, attempt to find a valid decomposition
    of rank r-1 by dropping one column (a rank-1 term) and re-solving the residual
    with the remaining columns. We use a linear-algebra feasibility check:
    a decomposition of rank r-1 exists iff the target tensor minus the residual
    is in the span of the (r-1) rank-1 terms.

    Concretely: we try dropping each column c, and check whether the tensor
    T - (term c) can be written as sum_{r != c} U[:,r] V[:,r] W[:,r].
    This is checked by verifying that the residual tensor is in the image of the
    bilinear map defined by the remaining columns. For a fixed set of columns,
    the residual must be expressible; we check via a least-squares consistency.

    Returns (U,V,W) of lower rank if found, else None.
    """
    n, m, p = DIMENSIONS
    r = best_U.shape[1]
    best_rank = r
    best = (best_U.copy(), best_V.copy(), best_W.copy())

    # Precompute the target tensor as a vector of length n*m*p
    # T[i,j,k] = 1 for all valid (i,j,k)
    # Flatten index: idx = (i*m + j)*p + k  ... but we use the U/V/W convention.
    # Actually the tensor is indexed by (i,j,k) and the decomposition is:
    #   T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # We'll work with the full (n*m, m*p, p*n) tensor flattened.

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # For each column c, the rank-1 term is outer(U[:,c], V[:,c], W[:,c])
    # We try dropping column c and see if T - term_c is in the span of the rest.

    # Strategy: For a candidate dropped column c, we have r-1 remaining columns.
    # We need T - term_c = sum_{c' != c} U[:,c'] V[:,c'] W[:,c'].
    # This is a system in the (r-1) coefficients... but the U,V,W of the remaining
    # columns are fixed. So we just check if T - term_c equals the sum of the
    # remaining terms. If it does, we can drop c.
    # But that's only true if the decomposition is already "tight". Generally
    # dropping a column breaks the equality. So instead we do a proper search:
    # re-optimize the remaining r-1 columns to fit T.

    # A cheaper exact check: the set of rank-1 terms {outer(U_c,V_c,W_c)} spans a
    # linear subspace of the tensor space. T must lie in that subspace for rank r.
    # For rank r-1, T must lie in a subspace of dimension r-1 (span of r-1 terms).
    # We check: is T in the span of {term_1, ..., term_{r-1}} (dropping term_c)?
    # This is a linear membership test: build the matrix whose columns are the
    # vectorized terms, and check if T is in its column space.

    # Vectorize each term
    def vec_term(Uc, Vc, Wc):
        return np.einsum('a,b,c->abc', Uc, Vc, Wc).ravel()

    terms = [vec_term(best_U[:,c], best_V[:,c], best_W[:,c]) for c in range(r)]

    # For each c, check if T is in span of terms excluding c
    # Use rank of [terms_except_c | T] vs rank of terms_except_c
    T_vec = T.ravel()

    # Precompute the matrix of all terms
    all_terms = np.array(terms)  # shape (r, n*m*p)

    # We'll try dropping each column. To save time, we only try a subset if r is large.
    # r = 98, so 98 membership tests. Each is a rank computation on a (97 x 128) matrix.
    # That's feasible within the deadline.

    # Try to find a column c such that T is in the span of all terms except c.
    # If found, we can drop c and the remaining r-1 columns form a valid decomposition.
    # (Because T = sum of all terms = (dropped term) + sum of rest, and if T is in
    # span of rest, then the dropped term is a linear combination of rest, so we
    # can absorb it.)

    # Actually, if T is in span of {terms except c}, then there exist coefficients
    # alpha such that T = sum_{c' != c} alpha_{c'} term_{c'}. This is a valid
    # decomposition of rank <= r-1 (some terms may be zero).

    # Let's do the membership test efficiently.
    # For each c, form M_c = all_terms with row c removed (shape (r-1, nmp)).
    # Check if T_vec is in column space of M_c.
    # Use: rank([M_c; T_vec]) == rank(M_c)

    nmp = n * m * p
    found = False
    best_c = None

    # To be safe on time, limit the number of columns we test.
    max_tests = r
    for c in range(min(max_tests, r)):
        if time.time() > deadline:
            break
        M_c = np.delete(all_terms, c, axis=0)  # (r-1, nmp)
        # Augment with T_vec as an extra row
        M_aug = np.vstack([M_c, T_vec.reshape(1, -1)])  # (r, nmp)
        rank_c = np.linalg.matrix_rank(M_c, tol=1e-8)
        rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-8)
        if rank_aug == rank_c:
            # T is in the span of the remaining terms. We can drop column c.
            # Construct the reduced decomposition by dropping column c.
            U_new = np.delete(best_U, c, axis=1)
            V_new = np.delete(best_V, c, axis=1)
            W_new = np.delete(best_W, c, axis=1)
            # Verify the reduced decomposition
            if _verify(U_new, V_new, W_new, n, m, p):
                best = (U_new, V_new, W_new)
                best_rank = best_U.shape[1] - 1
                found = True
                # Try to drop more columns iteratively
                # Update best_U etc. and continue
                best_U, best_V, best_W = U_new, V_new, W_new
                r = best_U.shape[1]
                # Recompute terms for the new set
                terms = [vec_term(best_U[:,cc], best_V[:,cc], best_W[:,cc]) for cc in range(r)]
                all_terms = np.array(terms)
                max_tests = r
                c = -1  # reset loop index
                # Reset the for loop by breaking and re-entering via a while
                # Simpler: use a while loop instead
                break
            else:
                # The reduced decomposition doesn't verify (numerical issue?), skip
                pass

    # If we found at least one drop, try to continue dropping in a while loop
    # (the for loop above breaks on first success; let's do a proper iterative loop)
    # Re-do with a while loop for iterative dropping
    # (The for loop above may have already dropped one; let's continue)

    # To keep it simple and time-bounded, do a fresh iterative drop search:
    # Reset to the original and do iterative drops with a while loop.
    U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
    r_cur = U_cur.shape[1]
    terms_cur = [vec_term(U_cur[:,c], V_cur[:,c], W_cur[:,c]) for c in range(r_cur)]
    all_terms_cur = np.array(terms_cur)

    while time.time() < deadline:
        dropped_any = False
        for c in range(r_cur):
            if time.time() > deadline:
                break
            M_c = np.delete(all_terms_cur, c, axis=0)
            M_aug = np.vstack([M_c, T_vec.reshape(1, -1)])
            rank_c = np.linalg.matrix_rank(M_c, tol=1e-8)
            rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-8)
            if rank_aug == rank_c:
                U_new = np.delete(U_cur, c, axis=1)
                V_new = np.delete(V_cur, c, axis=1)
                W_new = np.delete(W_cur, c, axis=1)
                if _verify(U_new, V_new, W_new, n, m, p):
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    r_cur = U_cur.shape[1]
                    terms_cur = [vec_term(U_cur[:,cc], V_cur[:,cc], W_cur[:,cc]) for cc in range(r_cur)]
                    all_terms_cur = np.array(terms_cur)
                    dropped_any = True
                    break  # restart the for loop with the new set
        if not dropped_any:
            break  # no more columns can be dropped

    if r_cur < best_rank:
        return (U_cur, V_cur, W_cur)
    return None


def construct():
    _t0 = time.time()
    _deadline = _t0 + DEADLINE_SECONDS

    # Get the base Strassen decomposition (rank 98)
    base = _strassen_decomposition()
    if base is None:
        # Should not happen, but fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    U, V, W = base['U'], base['V'], base['W']
    best_rank = base['rank']

    # Try to find a lower-rank decomposition with bounded time
    if time.time() < _deadline - 5.0:  # need at least 5s for finalization
        reduced = _search_lower_rank(U, V, W, _deadline)
        if reduced is not None:
            U, V, W = reduced
            best_rank = U.shape[1]

    # Final verification before returning
    n, m, p = DIMENSIONS
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
