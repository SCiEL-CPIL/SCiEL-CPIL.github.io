import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    # Build the known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_28(n, m, p)

    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j       # index in U
                b = j * p + k       # index in V
                c = k * n + i       # index in W (trace convention)
                T[a, b, c] = 1.0

    # Verify rank-28 solution
    U_arr = np.array(U28, dtype=float)
    V_arr = np.array(V28, dtype=float)
    W_arr = np.array(W28, dtype=float)
    max_err = _verify_decomp(U_arr, V_arr, W_arr, 28, T)
    if max_err > 1e-10:
        # Fallback to schoolbook
        U32, V32, W32 = _build_schoolbook_32(n, m, p)
        return {"U": U32, "V": V32, "W": W32}

    # Try to find rank-27 by checking linear dependencies
    if time.time() < deadline - 20:
        result = _search_rank_27(U_arr, V_arr, W_arr, T, deadline)
        if result is not None:
            U, V, W = result
            rank_found = U.shape[1]
            if _verify_decomp(U, V, W, rank_found, T) < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U28, "V": V28, "W": W28}

def _build_strassen_28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 matrix multiplication
    # Each term: (U_coeffs, V_coeffs, W_coeffs) where keys are local indices 0-3
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),        # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),        # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})         # M7
    ]

    # Blocks: (a_row_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4, so a_row_offset is 0 or 2
    # B is 4x4, so b_row_offset is 0 or 2, b_col_offset is 0 or 2
    # C is 2x4, so c_col_offset is 0 or 2
    blocks = [
        (0, 0, 0, 0),   # A_L * B_TL -> C_LL
        (2, 2, 0, 0),   # A_R * B_BL -> C_LR
        (0, 0, 2, 2),   # A_L * B_TR -> C_RL
        (2, 2, 2, 2)    # A_R * B_BR -> C_RR
    ]

    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in u_coeffs.items():
                # U index: i*m + j where i = local_idx//2, j = local_idx%2
                # But with offset: i = local_idx//2, j = a_off + local_idx%2
                g_idx = (local_idx // 2) * m + a_off + (local_idx % 2)
                U[g_idx][col_idx] = val
            for local_idx, val in v_coeffs.items():
                # V index: j*p + k where j = b_row_off + local_idx//2, k = b_col_off + local_idx%2
                g_idx = (b_row_off + local_idx // 2) * p + b_col_off + (local_idx % 2)
                V[g_idx][col_idx] = val
            for local_idx, val in w_coeffs.items():
                # W index: k*n + i where k = c_off + local_idx%2, i = local_idx//2
                g_idx = (c_off + local_idx % 2) * n + (local_idx // 2)
                W[g_idx][col_idx] = val
            col_idx += 1

    return U, V, W

def _build_schoolbook_32(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return U, V, W

def _verify_decomp(U_arr, V_arr, W_arr, rank_val, T):
    """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
    nm, mp, pn = T.shape
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank_val):
        S += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    return np.max(np.abs(S - T))

def _search_rank_27(U_arr, V_arr, W_arr, T, deadline):
    """Search for rank-27 by checking linear dependencies and cross-block merging."""
    rank = 28
    nm, mp, pn = T.shape

    # Build M matrix: M[abc, r] = U[a,r]*V[b,r]*W[c,r]
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r]).flatten()

    # Check overall rank of M
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        effective_rank = np.sum(s > 1e-8)
        if effective_rank < rank:
            # Find a basis of size effective_rank
            # Use QR with column pivoting to find independent columns
            Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
            # R is upper triangular; find first effective_rank columns
            U_new = U_arr[:, piv[:effective_rank]]
            V_new = V_arr[:, piv[:effective_rank]]
            W_new = W_arr[:, piv[:effective_rank]]
            return U_new, V_new, W_new
    except np.linalg.LinAlgError:
        pass

    # For each candidate term to eliminate, check if it's in the span of the others
    for r_elim in range(rank):
        if time.time() > deadline - 10:
            break

        other_cols = [s for s in range(rank) if s != r_elim]
        M_sub = M[:, other_cols]  # 1024 x 27
        rhs = M[:, r_elim]  # 1024

        # Solve least squares
        try:
            alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
            residual = np.max(np.abs(M_sub @ alpha - rhs))
            if residual < 1e-8:
                # Term r_elim is redundant!
                new_U = np.delete(U_arr, r_elim, axis=1)
                new_V = np.delete(V_arr, r_elim, axis=1)
                new_W = np.delete(W_arr, r_elim, axis=1)
                return new_U, new_V, new_W
        except np.linalg.LinAlgError:
            continue

    # Try to find a rank-27 decomposition by allowing linear combinations of terms
    # This is a harder problem; try a few random linear combinations
    if time.time() < deadline - 15:
        rng = np.random.RandomState(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Try to eliminate one term by expressing it as a combination of two others
            for r_elim in range(rank):
                # Pick two other terms
                others = [s for s in range(rank) if s != r_elim]
                if len(others) < 2:
                    continue
                r1, r2 = others[trial % len(others)], others[(trial + 1) % len(others)]
                # Check if M[:, r_elim] ≈ a*M[:, r1] + b*M[:, r2]
                A2 = M[:, [r1, r2]]
                rhs = M[:, r_elim]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A2, rhs, rcond=1e-10)
                    resid = np.max(np.abs(A2 @ coeffs - rhs))
                    if resid < 1e-8:
                        # We can eliminate r_elim
                        # But we need to adjust the other terms
                        # New term r1' = r1 - coeffs[0]*r_elim (but r_elim is gone)
                        # Actually, if r_elim = a*r1 + b*r2, then we just drop r_elim
                        new_U = np.delete(U_arr, r_elim, axis=1)
                        new_V = np.delete(V_arr, r_elim, axis=1)
                        new_W = np.delete(W_arr, r_elim, axis=1)
                        return new_U, new_V, new_W
                except np.linalg.LinAlgError:
                    continue

    return None
