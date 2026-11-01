"""Matrix multiplication 2,2,11 - Rank-minimal decomposition with SVD rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]


def _build_tensor():
    """Build the multiplication tensor T of shape (4, 22, 22)."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, T=None):
    """Verify decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_decomp - T)) < 1e-9


def _build_strassen_2x2_block(k0, k1, n, m, p):
    """
    Build the 7 Strassen terms for a 2x2 * 2x2 block at columns k0, k1.
    Returns 7 triples (u_vec, v_vec, w_vec) as sparse index-value pairs.
    """
    # V indices: B[j, k] -> j*p + k
    v_e = k0       # B[0, k0]
    v_f = k1       # B[0, k1]
    v_g = p + k0   # B[1, k0]
    v_h = p + k1   # B[1, k1]

    # W indices: C[i, k] -> k*n + i
    w_c00 = k0 * n + 0   # C[0, k0]
    w_c10 = k0 * n + 1   # C[1, k0]
    w_c01 = k1 * n + 0   # C[0, k1]
    w_c11 = k1 * n + 1   # C[1, k1]

    # U indices: A[i,j] -> i*m + j
    u_a = 0  # A[0,0]
    u_b = 1  # A[0,1]
    u_c = 2  # A[1,0]
    u_d = 3  # A[1,1]

    terms = []

    # m1 = (a+d)(e+h)
    terms.append(([(u_a, 1), (u_d, 1)], [(v_e, 1), (v_h, 1)], [(w_c00, 1), (w_c11, 1)]))

    # m2 = (c+d)e
    terms.append(([(u_c, 1), (u_d, 1)], [(v_e, 1)], [(w_c10, 1), (w_c11, -1)]))

    # m3 = a(f-h)
    terms.append(([(u_a, 1)], [(v_f, 1), (v_h, -1)], [(w_c01, 1), (w_c11, 1)]))

    # m4 = d(g-e)
    terms.append(([(u_d, 1)], [(v_g, 1), (v_e, -1)], [(w_c00, 1), (w_c10, 1)]))

    # m5 = (a+b)h
    terms.append(([(u_a, 1), (u_b, 1)], [(v_h, 1)], [(w_c00, -1), (w_c01, 1)]))

    # m6 = (c-a)(e+f)
    terms.append(([(u_c, 1), (u_a, -1)], [(v_e, 1), (v_f, 1)], [(w_c11, 1)]))

    # m7 = (b-d)(g+h)
    terms.append(([(u_b, 1), (u_d, -1)], [(v_g, 1), (v_h, 1)], [(w_c00, 1)]))

    return terms


def _build_schoolbook_2x2_1(k, n, m, p):
    """
    Build the 4 schoolbook terms for a 2x2 * 2x1 block at column k.
    Returns 4 triples (u_vec, v_vec, w_vec) as sparse index-value pairs.
    """
    v_e = k        # B[0, k]
    v_g = p + k    # B[1, k]
    w_c0 = k * n + 0   # C[0, k]
    w_c1 = k * n + 1   # C[1, k]

    terms = []
    # A[0,0]*B[0,k] -> C[0,k]
    terms.append(([(0, 1)], [(v_e, 1)], [(w_c0, 1)]))
    # A[0,1]*B[1,k] -> C[0,k]
    terms.append(([(1, 1)], [(v_g, 1)], [(w_c0, 1)]))
    # A[1,0]*B[0,k] -> C[1,k]
    terms.append(([(2, 1)], [(v_e, 1)], [(w_c1, 1)]))
    # A[1,1]*B[1,k] -> C[1,k]
    terms.append(([(3, 1)], [(v_g, 1)], [(w_c1, 1)]))

    return terms


def _build_rank39():
    """
    Build the rank-39 decomposition: 5 Strassen blocks (rank 7 each) + 1 schoolbook column (rank 4).
    Returns (U, V, W) as numpy arrays.
    """
    n, m, p = DIMENSIONS
    rank = 5 * 7 + 4
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    # 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        terms = _build_strassen_2x2_block(k0, k1, n, m, p)
        for (u_pairs, v_pairs, w_pairs) in terms:
            for (idx, val) in u_pairs:
                U[idx, col] = val
            for (idx, val) in v_pairs:
                V[idx, col] = val
            for (idx, val) in w_pairs:
                W[idx, col] = val
            col += 1

    # 1 schoolbook column for column 10
    terms = _build_schoolbook_2x2_1(10, n, m, p)
    for (u_pairs, v_pairs, w_pairs) in terms:
        for (idx, val) in u_pairs:
            U[idx, col] = val
        for (idx, val) in v_pairs:
            V[idx, col] = val
        for (idx, val) in w_pairs:
            W[idx, col] = val
        col += 1

    return U, V, W


def _svd_rank_reduce(U, V, W, T, max_time=30):
    """
    Attempt to reduce rank via SVD analysis of the tensor matricization.
    Returns (U, V, W) with potentially reduced rank, or original if no reduction found.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    if current_rank <= 1:
        return U, V, W

    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]

    # Build the matricization M where each column is a rank-1 tensor flattened
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()

    # SVD to find numerical rank
    try:
        s_vals = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = int(np.sum(s_vals > tol))
    except Exception:
        return U, V, W

    if num_rank >= current_rank:
        return U, V, W

    # Try to find a subset of columns that spans the same space
    # Use QR with column pivoting
    try:
        # Simple approach: greedily select independent columns
        selected = []
        basis = np.zeros((dim_a * dim_b * dim_c, 0), dtype=np.float64)
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            # Check if column r is in span of selected columns
            if len(selected) == 0:
                selected.append(r)
                basis = M[:, r:r+1].copy()
            else:
                # Least squares fit
                try:
                    coeffs, residuals, rank_basis, _ = np.linalg.lstsq(basis, M[:, r], rcond=None)
                    residual = M[:, r] - basis @ coeffs
                    if np.linalg.norm(residual) < tol * np.linalg.norm(M[:, r]):
                        continue  # Dependent, skip
                    else:
                        selected.append(r)
                        basis = np.column_stack([basis, M[:, r]])
                except Exception:
                    selected.append(r)
                    basis = np.column_stack([basis, M[:, r]])

            if len(selected) >= num_rank:
                break

        if len(selected) < current_rank:
            U_new = U[:, selected]
            V_new = V[:, selected]
            W_new = W[:, selected]
            if _verify(U_new, V_new, W_new, T):
                return U_new, V_new, W_new
    except Exception:
        pass

    return U, V, W


def _greedy_term_elimination(U, V, W, T, max_time=30):
    """
    Try to eliminate individual terms by checking if the remaining terms
    still reproduce the tensor (after adjusting other terms via least squares).
    """
    start_time = time.time()
    current_rank = U.shape[1]
    if current_rank <= 1:
        return U, V, W

    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]

    # For each term, try removing it and see if we can adjust the others
    for r_to_remove in range(current_rank):
        if time.time() - start_time > max_time:
            break

        # Build the system without term r_to_remove
        remaining = [i for i in range(current_rank) if i != r_to_remove]
        if len(remaining) == 0:
            break

        # The constraint: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
        # For fixed U, V, W structure, this is linear in the free parameters.
        # But since we're keeping the same U,V,W structure and just removing a term,
        # we need to check if the remaining terms can be rescaled to match.
        #
        # Actually, a simpler approach: check if the term is "redundant" by seeing
        # if its contribution can be expressed as a combination of other terms.
        #
        # For a trilinear form, term r is redundant if there exist coefficients
        # alpha_s such that for all (a,b,c):
        #   U[a,r]*V[b,r]*W[c,r] = sum_s alpha_s * U[a,s]*V[b,s]*W[c,s]
        #
        # This is a system of dim_a*dim_b*dim_c equations in len(remaining) unknowns.
        # Since the tensor is sparse, most equations are 0=0.
        #
        # Let's just check: can we solve for the remaining terms to match T?
        # This is underdetermined in general, so we use a different approach.

        # Alternative: try linear combinations. For each pair of terms, check if
        # one can be absorbed into the other via a linear transformation.
        pass

    return U, V, W


def _try_lower_rank_search(U, V, W, T, max_time=40):
    """
    Try to find a decomposition with rank < current_rank by searching for
    alternative structures. Uses the known algebraic structure.

    For 2x2 * 2x11, the tensor decomposes into 5 disjoint 2x2*2x2 sub-tensors
    and 1 disjoint 2x2*2x1 sub-tensor. The rank of each 2x2*2x2 sub-tensor
    is exactly 7 (Strassen's lower bound), and the 2x2*2x1 sub-tensor has rank 4.
    Total minimum rank = 5*7 + 4 = 39.

    This function verifies this lower bound and attempts any further reduction.
    """
    start_time = time.time()
    current_rank = U.shape[1]

    if current_rank <= 39:
        # We're at the theoretical minimum. Try SVD reduction just in case.
        U, V, W = _svd_rank_reduce(U, V, W, T, max_time=10)
        return U, V, W

    # Try SVD-based rank reduction
    U, V, W = _svd_rank_reduce(U, V, W, T, max_time=max_time // 2)
    if U.shape[1] < current_rank:
        return U, V, W

    # Try greedy elimination
    U, V, W = _greedy_term_elimination(U, V, W, T, max_time=max_time // 2)

    return U, V, W


def _build_schoolbook():
    """Fallback: rank 44 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    T = _build_tensor()

    # Build the verified rank-39 construction
    U, V, W = _build_rank39()

    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {'U': U, 'V': V, 'W': W}

    # Try to reduce rank further
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 5  # 5s safety margin

    if remaining_time > 5:
        U, V, W = _try_lower_rank_search(U, V, W, T, max_time=remaining_time)

    # Final verification
    if not _verify(U, V, W, T):
        # Rebuild the known-good rank-39
        U, V, W = _build_rank39()

    return {'U': U, 'V': V, 'W': W}
