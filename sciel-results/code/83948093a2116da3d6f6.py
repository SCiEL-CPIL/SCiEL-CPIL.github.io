import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Strategy: Build a decomposition and then try to reduce rank
    # using null space analysis and iterative column removal

    # Start with a naive decomposition (rank = 54)
    # Each term corresponds to one (am, bm) pair where am==bj
    rank_naive = m * p  # 27 terms... wait, let me reconsider

    # Actually, the naive decomposition: for each middle index am (0..2),
    # and for each column bm (0..8), we have one term
    # That gives 3*9 = 27 terms, but each term contributes to 2 outputs
    # Let me think more carefully.

    # The tensor T[i,j,k] = 1 iff i = am*m + ai, j = bm*p + am, k = bm*n + ai
    # for some am, ai, bm.
    # 
    # A rank-1 term is U[i]*V[j]*W[k]
    # For a fixed (am, bm, ai), the term is:
    #   U[am*m+ai] = 1, V[bm*p+am] = 1, W[bm*n+ai] = 1
    # This gives 3*9*2 = 54 terms for the naive decomposition.
    #
    # But we can do better by grouping.

    # Let me try the Strassen-based approach from the current program
    # and then attempt rank reduction.

    # Build rank-50 decomposition (same as current)
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        U[u_c, col+1] += 1
        U[u_d, col+1] += 1
        V[v_e, col+1] += 1

        U[u_a, col+2] += 1
        V[v_f, col+2] += 1
        V[v_h, col+2] -= 1

        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        V[v_e, col+3] -= 1

        U[u_a, col+4] += 1
        U[u_b, col+4] += 1
        V[v_h, col+4] += 1

        U[u_c, col+5] += 1
        U[u_a, col+5] -= 1
        V[v_e, col+5] += 1
        V[v_f, col+5] += 1

        U[u_b, col+6] += 1
        U[u_d, col+6] -= 1
        V[v_g, col+6] += 1
        V[v_h, col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] += 1

        W[w_01, col+2] += 1
        W[w_01, col+4] += 1

        W[w_10, col+1] += 1
        W[w_10, col+3] += 1

        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a, col] += 1
    V[v_e, col] += 1
    U[u_b, col+1] += 1
    V[v_g, col+1] += 1
    U[u_c, col+2] += 1
    V[v_e, col+2] += 1
    U[u_d, col+3] += 1
    V[v_g, col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08, col] += 1
    W[w_08, col+1] += 1
    W[w_18, col+2] += 1
    W[w_18, col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0, col] += 1
        V[v_b, col] += 1
        W[k * n + 0, col] += 1
        U[u_ar1, col+1] += 1
        V[v_b, col+1] += 1
        W[k * n + 1, col+1] += 1
        col += 2

    # Verify the decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        # Fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank by finding and removing linearly dependent columns
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Build the coefficient matrix for null space analysis
    n_entries = n * m * m * p * p * n
    mat = np.zeros((n_entries, rank))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        vec = np.einsum('i,j,k->ijk', u_r, v_r, w_r).flatten()
        mat[:, r] = vec

    # Compute SVD to find numerical rank
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)

        if num_rank < rank:
            null_dim = rank - num_rank

            # Try removing columns one at a time, checking if the decomposition still works
            # Use the null space vectors to identify candidates
            current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
            current_rank = rank

            # Iteratively try to remove columns
            for iteration in range(null_dim * 3):
                if time.time() > deadline - 15:
                    break

                # Find null space of current matrix
                mat_current = np.zeros((n_entries, current_rank))
                for r in range(current_rank):
                    u_r = current_U[:, r]
                    v_r = current_V[:, r]
                    w_r = current_W[:, r]
                    vec = np.einsum('i,j,k->ijk', u_r, v_r, w_r).flatten()
                    mat_current[:, r] = vec

                _, s_cur, Vt_cur = np.linalg.svd(mat_current, full_matrices=False)
                tol_cur = s_cur[0] * current_rank * 1e-10
                num_rank_cur = np.sum(s_cur > tol_cur)

                if num_rank_cur >= current_rank:
                    break

                null_dim_cur = current_rank - num_rank_cur

                # Get null space vectors
                null_vectors = Vt_cur[num_rank_cur:]

                # Find best column to remove
                best_drop = None
                best_residual = float('inf')

                for nv_idx in range(min(null_dim_cur, 5)):
                    null_vec = null_vectors[nv_idx]
                    abs_null = np.abs(null_vec)
                    threshold = np.max(abs_null) * 0.05
                    candidates = np.where(abs_null > threshold)[0]

                    for drop_col in candidates:
                        if time.time() > deadline - 10:
                            break

                        mask = np.ones(current_rank, dtype=bool)
                        mask[drop_col] = False
                        U_new = current_U[:, mask]
                        V_new = current_V[:, mask]
                        W_new = current_W[:, mask]

                        decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                        residual = np.sum((decomp_new - T)**2)

                        if residual < best_residual:
                            best_residual = residual
                            best_drop = drop_col

                        if residual < 1e-8:
                            break

                    if best_residual < 1e-8:
                        break

                if best_drop is None or best_residual > 1e-8:
                    break

                # Remove the column
                mask = np.ones(current_rank, dtype=bool)
                mask[best_drop] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1

                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

            if best_rank < rank:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
