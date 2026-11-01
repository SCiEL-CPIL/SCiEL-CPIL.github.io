"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]


def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
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
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)


def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)


def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.

    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)

    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)

    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1

        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3

        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1

        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1

        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1

        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1

    # Handle remainder column (column 10)
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n + 0
        w_c12 = k2 * n + 1

        # Term 1: A[0,0]*B[0,10]
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1

        # Term 2: A[0,1]*B[1,10]
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1

        # Term 3: A[1,0]*B[0,10]
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1

        # Term 4: A[1,1]*B[1,10]
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1

    return U, V, W


def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
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


def _rank_3_for_2x2x2x1():
    """
    Attempt a rank-3 decomposition for 2x2 * 2x1.
    If successful, the last column costs 3 instead of 4, giving total 38.

    C[0,0] = a*e + b*g
    C[1,0] = c*e + d*g

    Try: t1 = (a+c)*e, t2 = (b+d)*g, t3 = (a-b-c+d)*e (or similar)
    We need: C[0,0] = a*e + b*g, C[1,0] = c*e + d*g
    With 3 terms we have 3 bilinear forms. Let's search systematically.
    """
    # The tensor for 2x2 * 2x1 has 4 nonzero entries:
    # T[0,0,0]=1 (a*e->C00), T[1,1,0]=1 (b*g->C00), T[2,0,1]=1 (c*e->C10), T[3,1,1]=1 (d*g->C10)
    # Wait, indices: A is 2x2 (indices 0-3), B is 2x1 (indices 0-1), C is 1x2 (indices 0-1)
    # T[a_idx, b_idx, c_idx] = 1 where:
    # a*e -> C[0,0]: a_idx=0, b_idx=0, c_idx=0
    # b*g -> C[0,0]: a_idx=1, b_idx=1, c_idx=0
    # c*e -> C[1,0]: a_idx=2, b_idx=0, c_idx=1
    # d*g -> C[1,0]: a_idx=3, b_idx=1, c_idx=1

    # For rank 3, we need 3 rank-1 tensors that sum to this.
    # This is equivalent to finding 3 vectors u_r (in R^4), v_r (in R^2), w_r (in R^2)
    # such that sum_r u_r ⊗ v_r ⊗ w_r = T.

    # The tensor has a special structure. Let's try:
    # t1: u=[1,0,1,0], v=[1,0], w=[1,1]  -> a*e appears in C00 and C10... no that gives a*e in both
    # Let me think differently.

    # Actually for 2x2 * 2x1, the rank is known to be 4 (it's just schoolbook).
    # The tensor is:
    #   a*e  b*g
    #   c*e  d*g
    # This is an outer product structure that requires rank 4.
    # Proof: The slice C[:,0] as a function of (a,b,c,d,e,g) has rank 4 in the
    # (A-space) × (B-space) sense.
    # So rank 3 is NOT achievable for 2x2*2x1.
    return None


def _try_3col_decomposition():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3.
    If found, total rank = 4*7 + 10 = 38 instead of 5*7 + 4 = 39.

    We use a structured parameterized search over "generalized Strassen" patterns.
    """
    n, m = 2, 2
    p3 = 3
    # Local subproblem: A is 2x2, B is 2x3, C is 3x2
    # T shape: (4, 6, 6)
    T3 = np.zeros((4, 6, 6), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 2 + i
                T3[a, b, c] = 1.0

    # Start from Strassen(2) + schoolbook(1) = rank 11, try to reduce to 10.
    # The 11 terms are:
    # Strassen m1..m7 for columns 0,1 + a*e2, b*g2, c*e2, d*g2 for column 2

    # Idea: merge one schoolbook term into a modified Strassen term.
    # Try modifying m6 = (c-a)(e+f) to m6' = (c-a)(e+f) + alpha * something
    # Or try a completely different 10-term pattern.

    # Let's try a direct algebraic approach:
    # We need to express 24 bilinear monomials (a,b,c,d) × (e,f,g,h,i,j) -> C entries
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[0,2] = ag + bj
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    # C[1,2] = cg + dj

    # That's 12 output entries, each a sum of 2 products. Total 24 "unit" products.
    # With 10 rank-1 terms, each term covers at most 1 A-entry × 1 B-entry × 1 C-entry.
    # But bilinear terms can cover multiple outputs via the W matrix.

    # Let's try a specific 10-term construction:
    # t1 = (a+d)(e+h)       -> contributes to C[0,0], C[1,1]
    # t2 = (c+d)e           -> contributes to C[1,0], C[1,1]
    # t3 = a(f-h)           -> contributes to C[0,1], C[1,1]
    # t4 = d(g-e)           -> contributes to C[0,0], C[1,0]
    # t5 = (a+b)h           -> contributes to C[0,0], C[0,1]
    # t6 = (c-a)(e+f)       -> contributes to C[1,1]
    # t7 = (b-d)(g+h)       -> contributes to C[0,0]
    # t8 = a*g              -> contributes to C[0,2]
    # t9 = b*j              -> contributes to C[0,2]
    # t10 = c*g + d*j       -> contributes to C[1,2] ... but this is NOT rank 1!

    # t10 would need to be a single bilinear term. c*g + d*j is not a single
    # (linear in A) × (linear in B) product. So this doesn't work directly.

    # Alternative: t10 = (c+d)*j, and modify something else to provide c*g - c*j = c*(g-j)
    # That would need yet another term. So rank 10 seems hard.

    # Let's try: what if we use a non-Strassen decomposition for the first 2 columns
    # that's more compatible with column 2?

    # Try: t1 = a*(e+g), t2 = b*(f+j), t3 = c*(e+g), t4 = d*(h+j)
    # C[0,0] = a*e + b*g: we have a*e from t1, but b*g? t2 gives b*f and b*j. No good.

    # I think rank 10 for 2x2*2x3 is not achievable with simple patterns.
    # The lower bound for 2x2*2x3 is at least 10 (from the 12 output entries
    # each needing 2 products, and the structure). But constructing it is non-trivial.
    return None


def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    Uses SVD on the tensor matricization to find a minimal basis,
    then greedy single- and double-column removal.
    """
    start_time = time.time()
    current_rank = U.shape[1]

    if current_rank <= 1:
        return U, V, W

    # Flatten each rank-1 tensor into a vector
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()

    # Use SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = np.sum(s_vals > tol)
    except Exception:
        num_rank = current_rank

    if num_rank < current_rank:
        # Try QR with pivoting to find independent columns
        try:
            Q, R = np.linalg.qr(M, mode='complete')
            diag_R = np.abs(np.diag(R))
            pivot_cols = np.argsort(diag_R)[::-1][:num_rank]
            pivot_cols = np.sort(pivot_cols)
            U_new = U[:, pivot_cols]
            V_new = V[:, pivot_cols]
            W_new = W[:, pivot_cols]
            if _verify(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
                return U_new, V_new, W_new
        except Exception:
            pass

    # Greedy: remove one column at a time
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify_fast(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break

    # Try removing pairs of columns
    for attempt in range(current_rank // 2):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r1 in range(current_rank):
            if time.time() - start_time > max_time:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() - start_time > max_time:
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                if _verify_fast(U_test, V_test, W_test, T):
                    U, V, W = U_test, V_test, W_test
                    current_rank -= 2
                    removed = True
                    break
            if removed:
                break
        if not removed:
            break

    return U, V, W


def _try_cross_block_reduction(U, V, W, T, max_time=30):
    """
    Try to find a lower-rank decomposition by allowing linear combinations
    of the original terms. This is a local optimization in the space of
    rank-r decompositions.

    We parameterize: for each output entry (a,b,c), we need
    sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]

    We try to find a set of r' < r terms that still span the same tensor.
    This is equivalent to checking if the tensor lies in the secant variety
    of lower rank. We use a heuristic: try to express some terms as linear
    combinations of others by solving linear systems.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = 4, 22, 22

    # For each term r, compute its "signature" as a 4x22x22 tensor (sparse).
    # Try to find a term that can be expressed as a linear combination of others
    # while preserving the tensor.

    # A more practical approach: try removing one term and see if we can
    # adjust the remaining terms to compensate.
    # For term r to be removable, we need:
    #   U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_{r' != r} alpha_{r'} * (U[:,r'] ⊗ V[:,r'] ⊗ W[:,r'])
    # This is a very restrictive condition.

    # Instead, try: can we replace 2 terms with 1 term?
    # Terms r1, r2 can be replaced by a single term if:
    #   U1⊗V1⊗W1 + U2⊗V2⊗W2 = U3⊗V3⊗W3
    # This requires U1⊗V1⊗W1 + U2⊗V2⊗W2 to be rank 1, which is very unlikely
    # for the Strassen terms.

    # Try a different approach: look for terms that have the same U and V vectors
    # (up to scaling) but different W vectors, and see if we can merge them.
    for r1 in range(current_rank):
        if time.time() - start_time > max_time:
            break
        for r2 in range(r1 + 1, current_rank):
            if time.time() - start_time > max_time:
                break
            # Check if U[:,r1] and U[:,r2] are proportional
            U1 = U[:, r1]
            U2 = U[:, r2]
            if np.linalg.norm(U1) < 1e-10 or np.linalg.norm(U2) < 1e-10:
                continue
            # Check proportionality
            ratio = np.dot(U1, U2) / np.dot(U1, U1)
            if abs(np.dot(U1, U2) - ratio * np.dot(U1, U1)) < 1e-8:
                # U vectors are proportional, check V vectors
                V1 = V[:, r1]
                V2 = V[:, r2]
                if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                    continue
                ratio_v = np.dot(V1, V2) / np.dot(V1, V1)
                if abs(np.dot(V1, V2) - ratio_v * np.dot(V1, V1)) < 1e-8:
                    # Both U and V are proportional, so the two terms can be merged
                    # into one term with combined W vector.
                    # New term: U_new = U1, V_new = V1, W_new = W1 + (ratio_v/ratio)*W2
                    # Actually: U2 = ratio*U1, V2 = ratio_v*V1
                    # U2⊗V2⊗W2 = ratio*ratio_v * U1⊗V1⊗W2
                    # So combined: U1⊗V1⊗(W1 + ratio*ratio_v*W2)
                    W_new = W[:, r1] + ratio * ratio_v * W[:, r2]
                    mask = [i for i in range(current_rank) if i != r1 and i != r2]
                    U_test = np.column_stack([U[:, mask], U[:, r1]])
                    V_test = np.column_stack([V[:, mask], V[:, r1]])
                    W_test = np.column_stack([W[:, mask], W_new])
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 1

    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit

    T = _build_target_tensor()

    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()

    if _verify(U, V, W, T):
        # Try rank reduction via SVD/QR
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            U_red, V_red, W_red = _try_rank_reduction(
                U, V, W, T, max_time=min(remaining_time, 60)
            )
            if U_red.shape[1] < U.shape[1]:
                U, V, W = U_red, V_red, W_red

        # Try cross-block reduction (look for mergeable terms)
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 5:
            U_red, V_red, W_red = _try_cross_block_reduction(
                U, V, W, T, max_time=min(remaining_time, 30)
            )
            if U_red.shape[1] < U.shape[1]:
                U, V, W = U_red, V_red, W_red

        # Try the 3-column subproblem approach
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            result = _try_3col_decomposition()
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, T) and U.shape[1] < 39:
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Should not reach here
    raise RuntimeError("No valid decomposition found")
