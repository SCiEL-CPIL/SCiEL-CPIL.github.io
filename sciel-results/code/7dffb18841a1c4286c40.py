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
    """
    start_time = time.time()
    current_rank = U.shape[1]
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = 4, 22, 22

    # Look for terms that have the same U and V vectors (up to scaling)
    # but different W vectors, and see if we can merge them.
    for r1 in range(current_rank):
        if time.time() - start_time > max_time:
            break
        for r2 in range(r1 + 1, current_rank):
            if time.time() - start_time > max_time:
                break
            U1 = U[:, r1]
            U2 = U[:, r2]
            if np.linalg.norm(U1) < 1e-10 or np.linalg.norm(U2) < 1e-10:
                continue
            ratio = np.dot(U1, U2) / np.dot(U1, U1)
            if abs(np.dot(U1, U2) - ratio * np.dot(U1, U1)) < 1e-8:
                V1 = V[:, r1]
                V2 = V[:, r2]
                if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                    continue
                ratio_v = np.dot(V1, V2) / np.dot(V1, V1)
                if abs(np.dot(V1, V2) - ratio_v * np.dot(V1, V1)) < 1e-8:
                    W_new = W[:, r1] + ratio * ratio_v * W[:, r2]
                    mask = [i for i in range(current_rank) if i != r1 and i != r2]
                    U_test = np.column_stack([U[:, mask], U[:, r1]])
                    V_test = np.column_stack([V[:, mask], V[:, r1]])
                    W_test = np.column_stack([W[:, mask], W_new])
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 1

    return U, V, W


def _try_3col_rank10():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3.
    
    The 2x2 * 2x3 problem has 12 output entries, each a sum of 2 products.
    Total 24 unit products. With 10 rank-1 terms, we need clever sharing.
    
    Let's try a systematic search over parameterized families.
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

    # Try various 10-term constructions
    # The key idea: can we share terms between the 2x2 Strassen block and the 3rd column?
    
    # Construction attempt:
    # Use 7 Strassen terms for columns 0,1
    # Plus 3 terms for column 2, but modify some Strassen terms to help
    
    # Let's try:
    # t1-t7: standard Strassen for cols 0,1
    # t8: a*(e+g2) where g2 = B[0,2]
    # t9: b*(f+h2) where h2 = B[1,2]  
    # t10: (c+d)*(e+g2) - but this doesn't work cleanly
    
    # Alternative: try a completely different 10-term pattern
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[0,2] = ag + bj
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    # C[1,2] = cg + dj
    
    # Try:
    # s1 = (a+d)(e+h)     -> C[0,0] += 1, C[1,1] += 1
    # s2 = (c+d)e         -> C[1,0] += 1, C[1,1] -= 1
    # s3 = a(f-h)         -> C[0,1] += 1, C[1,1] += 1
    # s4 = d(g-e)         -> C[0,0] += 1, C[1,0] += 1
    # s5 = (a+b)h         -> C[0,0] -= 1, C[0,1] += 1
    # s6 = (c-a)(e+f)     -> C[1,1] += 1
    # s7 = (b-d)(g+h)     -> C[0,0] += 1
    # s8 = a*g            -> C[0,2] += 1
    # s9 = b*j            -> C[0,2] += 1
    # s10 = c*g + d*j     -> NOT rank 1!
    
    # So we need: c*g and d*j separately, but that's 2 more terms = 12 total.
    # Unless we can modify existing terms.
    
    # What if s10 = (c+d)*j, and we adjust s4 to provide c*(g-j)?
    # s4' = d*(g-e) + c*(g-j) = d*g - d*e + c*g - c*j
    # That's not rank 1 either.
    
    # What if we use: s8 = a*(g), s9 = b*(j), s10 = (c+d)*(g) ... no
    # s10 = c*g + d*j is the problem.
    
    # Try: s10 = (c-d)*g + d*(g+j) ... no, that's 2 terms.
    
    # I think rank 10 for 2x2*2x3 is not achievable with simple patterns.
    # Let's verify: the lower bound. The tensor has 12 nonzero entries in C-space,
    # each requiring 2 products. The A-space has 4 entries, B-space has 6.
    # The rank must be at least max(4, 6, 12/2) = 6, but structural constraints
    # likely push it higher.
    
    # For now, return None (no improvement found)
    return None


def _try_aggressive_merging(U, V, W, T, max_time=45):
    """
    Try more aggressive merging: look for pairs of terms where one can be
    expressed as a linear combination of the other plus a third term,
    effectively reducing rank by 1.
    
    Also try: can we replace 3 terms with 2?
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 2:
        return U, V, W
    
    # For each pair (r1, r2), check if their sum is rank-1
    # i.e., U1⊗V1⊗W1 + U2⊗V2⊗W2 = U3⊗V3⊗W3
    # This requires the 4x22x22 tensor to have rank 1.
    
    # A faster check: if U1 and U2 are linearly independent, and V1 and V2
    # are linearly independent, then the sum is generally not rank-1.
    # But if one of them is proportional, we can merge.
    
    # Try triple merging: can terms r1, r2, r3 be replaced by 2 terms?
    # This is harder. Let's try a different approach.
    
    # Look for terms that share the same U vector (exactly, not just proportional)
    # and see if their V⊗W parts can be combined.
    from collections import defaultdict
    
    u_groups = defaultdict(list)
    for r in range(current_rank):
        U_r = U[:, r]
        # Use a hashable representation
        key = tuple(np.round(U_r, 8))
        u_groups[key].append(r)
    
    for key, indices in u_groups.items():
        if len(indices) < 2:
            continue
        # All these terms have the same U vector
        # Check if any pair has proportional V vectors
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                r1, r2 = indices[i], indices[j]
                V1 = V[:, r1]
                V2 = V[:, r2]
                if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                    continue
                ratio_v = np.dot(V1, V2) / np.dot(V1, V1)
                if abs(np.dot(V1, V2) - ratio_v * np.dot(V1, V1)) < 1e-8:
                    # V vectors are proportional, merge
                    W_new = W[:, r1] + ratio_v * W[:, r2]
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
        
        # Try aggressive merging
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 5:
            U_red, V_red, W_red = _try_aggressive_merging(
                U, V, W, T, max_time=min(remaining_time, 45)
            )
            if U_red.shape[1] < U.shape[1]:
                U, V, W = U_red, V_red, W_red

        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Should not reach here
    raise RuntimeError("No valid decomposition found")
