import numpy as np
import time

DIMENSIONS = [2, 4, 7]


def _build_schoolbook_rank56():
    """Build the straightforward schoolbook decomposition (rank 56)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U[i * m + j, k] = 1
                V[j * p + l, k] = 1
                W[l * n + i, k] = 1
                k += 1
    assert k == rank
    return U, V, W


def _build_rank50_block():
    """Current known rank-50 decomposition (6 Strassen + 2 schoolbook-221)."""
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank
    return U, V, W


def _verify(U, V, W):
    """Check that U, V, W produce the 2x4x7 multiplication tensor."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(Uf.shape[1]):
                    actual += Uf[i * m + j, r] * Vf[j * p + k, r] * Wf[k * n + i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W):
    """Vectorized verification."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    # Build T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_ = Uf.reshape(n, m, R)
    V_ = Vf.reshape(m, p, R)
    W_ = Wf.reshape(p, n, R)
    T = np.zeros((n, m, p))
    for r in range(R):
        T += U_[:, :, r][:, None, :] * V_[:, :, r][None, :, :] * W_[:, :, r][None, :, :]
    return np.all(np.abs(T - 1.0) < 1e-9)


def _gaussian_eliminate_columns(A):
    """Find a basis for the column space of A (rows=features, cols=vectors).
    Returns list of column indices that form a basis."""
    M, N = A.shape
    if N == 0:
        return []
    rows = M
    cols = N
    pivot_cols = []
    # Work on a copy
    B = A.astype(np.float64).copy()
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # find pivot
        idx = np.argmax(np.abs(B[r:, c]))
        if abs(B[r + idx, c]) < 1e-12:
            continue
        # swap
        B[[r, r + idx], :] = B[[r + idx, r], :]
        pivot_cols.append(c)
        # eliminate below
        B[r + 1:, c] -= B[r + 1:, r] * B[r, c]
        r += 1
    return pivot_cols


def _find_dependent_columns(A, tol=1e-10):
    """Find columns of A (M x K) that are linear combinations of earlier ones.
    Returns list of (index, coeffs_dict) where coeffs_dict maps earlier pivot col index -> scalar."""
    M, K = A.shape
    B = A.astype(np.float64).copy()
    pivot_rows = []  # (col_idx, row)
    dependent = []
    r = 0
    for c in range(K):
        if r >= M:
            # All remaining are dependent on all pivots
            # Just skip
            continue
        # Find pivot in column c starting at row r
        idx = r
        max_abs = 0.0
        for i in range(r, M):
            if abs(B[i, c]) > max_abs:
                max_abs = abs(B[i, c])
                idx = i
        if max_abs < tol:
            # column c is dependent on previous pivots
            dependent.append(c)
            continue
        # swap rows
        B[[r, idx], :] = B[[idx, r], :]
        pivot_rows.append((c, r))
        # eliminate below
        B[r + 1:, c] -= B[r + 1:, r] * B[r, c]
        r += 1
    return dependent


def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    R0 = n * m * p  # 56

    # Step 1: Build schoolbook rank-56 decomposition
    U, V, W = _build_schoolbook_rank56()

    # Step 2: Check current rank-50 for reference
    best_U, best_V, best_W = _build_rank50_block()
    best_rank = 50
    # Verify best works
    if not _verify_fast(best_U, best_V, best_W):
        # fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = 56

    # Step 3: LLL-inspired basis reduction on the tensor space
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # A rank-1 term (u, v, w) contributes outer(u, v, w) to T.
    # We want to express T as sum of rank-1 terms with small rank.
    #
    # Approach: work in the space of "partial products".
    # The multiplication tensor can be seen as: for each (i,j), we need
    # T[i,j,:] = sum_k' U[i*m+j, r_k'] * V[j*p+*, r_k'] * W[*, r_k']
    #
    # Key idea: consider the bilinear structure. For fixed j, the tensor
    # slice is: T[:, j, :] (shape n x p) = sum over "terms" of (a vector in R^n)
    # outer (a vector in R^p), where the vectors come from U[:, j] and W[:, j]
    # combined with V's row j.
    #
    # More precisely: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let a_r = U[:, j*m... wait, U is indexed by (i*m+j), so U[i*m+j, r] for fixed j
    # gives a vector in R^n (over i). Similarly W[k*n+i, r] for fixed j? No, W
    # doesn't depend on j. V[j*p+k, r] for fixed j gives a vector in R^p (over k).
    #
    # So: T[:,j,:] (n x p) = sum_r  u_r(j)  outer  w_r  ... no that's not right either.
    # Let me re-derive:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r (U[:, j*m ... no. U row index is i*m+j. For fixed j, U[i*m+j, :] is a
    #   vector in R^R indexed by i... no, it's a single row of U for each i.
    #
    # Let me think differently. The full tensor T is in R^{n x m x p}.
    # A rank-1 decomposition term r contributes:
    #   T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a rank-1 tensor in the (i,j,k) space ONLY if U, V, W are separable,
    # which they generally are not. So each "column" r of (U,V,W) defines a tensor
    # of the form: for each i,j,k, the product of three specific entries.
    #
    # This is NOT a standard rank-1 tensor decomposition in the i,j,k space.
    # It's a "matrix multiplication tensor" decomposition.
    #
    # The standard approach: the multiplication tensor for m1 x m2 by m2 x m3 has
    # rank r if there exist r triples (u_t, v_t, w_t) with u_t in R^{m1*m2},
    # v_t in R^{m2*m3}, w_t in R^{m1*m3} such that for all i,j,k:
    #   sum_t u_t[i*m2+j] * v_t[j*m3+k] * w_t[i*m3+k] ... wait, w_t[k*m1+i]?
    #
    # The convention here is W[k*n+i, r], so:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is the standard matrix multiplication tensor with the given index convention.
    #
    # For LLL-style compression: we can try to find linear dependencies among
    # the "rank-1 components" by viewing the problem in a transformed space.
    #
    # Practical approach: Try random combinations of rank-1 terms (with small
    # integer coefficients) and check if they sum to T. This is a search problem.
    #
    # Given the time constraint, let's try a targeted search:
    # 1. Start with rank-50
    # 2. Try to remove one term and see if the remaining 49 can be adjusted (by
    #    adding a linear combination) to still produce T.
    # 3. More systematically: try rank-49 by searching for a set of 49 terms.

    # Given the 120s constraint and the difficulty of this search, let's try
    # a few quick things:

    # Attempt 1: Check if there's a rank-49 by trying to eliminate one column
    # from the rank-50 and solving for the remaining.
    # This is a constrained problem that's hard to solve exactly.

    # Attempt 2: Use the structure. The tensor T[i,j,k]=1 for all i,j,k
    # (2x4x7 = 56 entries, all 1).
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all 56 (i,j,k).
    #
    # Lower bound: The tensor has "slice rank" considerations.
    # For fixed j, T[:,j,:] is the all-ones 2x7 matrix, which has rank 1.
    # But the constraint couples all j's through the V matrix.
    #
    # Known results: rank(2x2x2) = 7 (Strassen). rank(2x4x4) = ?
    # If rank(2x2x2)=7, then rank(2x4x4) <= 4*7 = 28 by blocks, but can do better.
    # rank(2x4x4) = 20 is known (Winograd). Actually, let me think...
    #
    # For 2x4x7: we can think of it as 2x4x(2+2+2+1) = sum of 2x4x2 (3 times) + 2x4x1
    # rank(2x4x2): This is multiplying 2x4 by 4x2. 
    #   = sum of two 2x2x2 blocks? No, it's one 2x4x2.
    #   rank(2x4x2) = ? Each 2x4x2 is equivalent to two 2x2x1 matrices? No.
    #   A (2x4) * B (4x2) = C (2x2).
    #   Split A = [A1 A2] where A1, A2 are 2x2. B = [B1; B2] where B1, B2 are 2x2.
    #   C = A1*B1 + A2*B2. This is sum of two 2x2x2 multiplications.
    #   rank(2x4x2) <= 2 * rank(2x2x2) = 2*7 = 14.
    #   Can we do better? The tensor 2x4x2 has 56... no, 2*4*2=16 entries.
    #   Lower bound: each of the 4 columns of B contributes independently...
    #   Actually, rank(2x4x2) = 14 is conjectured/known.
    #
    # So rank(2x4x7) <= 3*14 + rank(2x4x1) = 42 + 8 = 50.
    # This matches our current solution!
    #
    # Can we do better than 50? Only if rank(2x4x2) < 14 or rank(2x4x1) < 8,
    # or if there's sharing between blocks.
    # rank(2x4x1): A (2x4) * B (4x1) = C (2x1).
    #   = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x1.
    #   Each 2x2x1 has rank 4 (no known improvement).
    #   rank(2x4x1) <= 8. Lower bound: 2x4x1 has 8 entries, and each is independent...
    #   Actually rank(2x2x1) = 4 is proven. So rank(2x4x1) >= 2*4 = 8? 
    #   Hmm, not necessarily, because the two 2x2 blocks share the A columns.
    #   But I believe rank(2x4x1) = 8.
    #
    # So 50 seems to be the natural upper bound from block decomposition.
    # The question is whether sharing between the three 2x4x2 blocks can reduce it.
    #
    # Given the difficulty, let's try a quick random search for rank 49.

    # Quick random search for lower rank
    import random
    rng = random.Random(42)

    def try_random_rank(target_rank, num_trials=200):
        """Try to find a decomposition of given rank by random search."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            # Generate random small integer matrices
            U_r = np.zeros((n * m, target_rank), dtype=np.int64)
            V_r = np.zeros((m * p, target_rank), dtype=np.int64)
            W_r = np.zeros((p * n, target_rank), dtype=np.int64)
            for r in range(target_rank):
                # Sparse random entries
                for _ in range(rng.randint(2, 5)):
                    U_r[rng.randint(0, n*m-1), r] = rng.choice([-1, 1])
                for _ in range(rng.randint(2, 5)):
                    V_r[rng.randint(0, m*p-1), r] = rng.choice([-1, 1])
                for _ in range(rng.randint(2, 5)):
                    W_r[rng.randint(0, p*n-1), r] = rng.choice([-1, 1])
            if _verify_fast(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    # Try a few random searches for rank 49 (quick, limited)
    # This is unlikely to succeed but cheap to try
    if time.time() < deadline - 20:
        result = try_random_rank(49, num_trials=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 49

    # Return the best found
    if time.time() > deadline - 5:
        # Safety: return what we have
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
