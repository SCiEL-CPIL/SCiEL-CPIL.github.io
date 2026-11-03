def construct():
    import time
    import numpy as np

    _t0 = time.time()
    _DEADLINE = 148.0

    n, m, p = 3, 4, 6

    def _time_ok(margin=5.0):
        return time.time() - _t0 < _DEADLINE - margin

    def build_strassen_hybrid():
        """Schoolbook row 0, Strassen 2x2x2 for rows 1-2 with column pairs."""
        terms = []

        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))

        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)

        for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)

        return terms

    def build_strassen_hybrid_v2():
        """Strassen rows 0-1, schoolbook row 2."""
        terms = []

        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))

        for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_block(0, 1, 0, 1, k1, k2)
            add_strassen_block(0, 1, 2, 3, k1, k2)

        for j in range(4):
            for k in range(6):
                add_schoolbook_term(2, j, k)

        return terms

    def build_full_schoolbook():
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W

    def verify_full(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_tensor_matrix(U, V, W):
        """Build matrix M of shape (n*m*p, rank) where M[idx, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
        rank = U.shape[1]
        M = np.zeros((n * m * p, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        M[idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    idx += 1
        return M

    def find_null_space(M, tol_factor=1e-8):
        """Find null space vectors of M via SVD."""
        try:
            M_s, M_sv, M_vt = np.linalg.svd(M, full_matrices=False)
        except:
            return []
        tol_m = tol_factor * max(M_s.shape[0], 1) * (M_s[0] if len(M_s) > 0 else 1.0)
        num_pivot = sum(1 for s in M_s if s > tol_m)
        if num_pivot >= M_vt.shape[0]:
            return []
        return M_vt[num_pivot:]

    def reduce_by_scalar_proportionality(U, V, W):
        """Find and eliminate scalar-proportional column triples."""
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        active = list(range(rank))
        eliminated = 0

        for _ in range(100):
            if not _time_ok(20):
                break
            found = False
            for idx_a in range(len(active)):
                if found:
                    break
                ra = active[idx_a]
                ua = U[:, ra]
                if np.allclose(ua, 0, atol=1e-12):
                    continue
                for idx_b in range(idx_a + 1, len(active)):
                    rb = active[idx_b]
                    ub = V[:, rb]
                    if np.allclose(ub, 0, atol=1e-12):
                        continue
                    # Check if U[:,ra] is proportional to U[:,rb]
                    ub_u = U[:, rb]
                    if np.allclose(ub_u, 0, atol=1e-12):
                        continue
                    nz_ub_u = np.nonzero(ub_u)[0]
                    if len(nz_ub_u) == 0:
                        continue
                    pos = nz_ub_u[0]
                    if abs(ub_u[pos]) < 1e-12:
                        continue
                    c_u = ua[pos] / ub_u[pos]
                    if abs(c_u) < 1e-12:
                        continue
                    if not np.allclose(ua, c_u * ub_u, atol=1e-10):
                        continue
                    # Check V
                    va = V[:, ra]
                    vb = V[:, rb]
                    if np.allclose(vb, 0, atol=1e-12):
                        continue
                    nz_vb = np.nonzero(vb)[0]
                    if len(nz_vb) == 0:
                        continue
                    posv = nz_vb[0]
                    if abs(vb[posv]) < 1e-12:
                        continue
                    c_v = va[posv] / vb[posv]
                    if abs(c_v) < 1e-12:
                        continue
                    if not np.allclose(va, c_v * vb, atol=1e-10):
                        continue
                    # Check W
                    e_needed = 1.0 / (c_u * c_v)
                    wa = W[:, ra]
                    wb = W[:, rb]
                    if np.allclose(wa, e_needed * wb, atol=1e-10):
                        U[:, ra] = 0
                        V[:, ra] = 0
                        W[:, ra] = 0
                        active.remove(ra)
                        eliminated += 1
                        found = True
                        break
            if not found:
                break

        return U, V, W, eliminated

    def reduce_by_linear_combination(U, V, W):
        """
        Try to eliminate columns using linear dependencies in the tensor matrix.
        If we find a null vector c of the tensor matrix M, then
        sum_r c[r] * U[:,r]*V[:,r]*W[:,r] = 0.
        This doesn't directly allow elimination unless we can absorb terms.
        However, if one column's U is a linear combination of others,
        we might be able to redistribute.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        eliminated = 0

        if not _time_ok(25):
            return U, V, W, eliminated

        # Build tensor matrix
        M = build_tensor_matrix(U, V, W)
        null_vecs = find_null_space(M)

        for nv in null_vecs:
            if not _time_ok(25):
                break
            max_r = np.argmax(np.abs(nv))
            if abs(nv[max_r]) < 1e-10:
                continue

            # Try to eliminate column max_r
            # We need to show that U[:,max_r]*V[:,max_r]*W[:,max_r] can be
            # expressed as a combination of other terms
            # This is only possible if the "direction" of this term in tensor space
            # is already covered by other terms
            # For now, this is a placeholder - the scalar proportionality check above
            # handles the main case
            pass

        return U, V, W, eliminated

    def reduce_by_column_dependence(U, V, W):
        """
        Try to find columns where U[:,r1] is a linear combination of other U columns,
        and similarly for V and W, allowing redistribution.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        eliminated = 0

        if not _time_ok(30):
            return U, V, W, eliminated

        # For each column, check if its U vector is in the span of others
        # This is expensive, so only try for small rank
        if rank > 80:
            return U, V, W, eliminated

        active = [c for c in range(rank) if not np.allclose(U[:, c], 0, atol=1e-12)]
        if len(active) < 3:
            return U, V, W, eliminated

        for target in list(active):
            if not _time_ok(30):
                break
            others = [c for c in active if c != target]
            if len(others) < 2:
                continue

            U_others = U[:, others]
            U_target = U[:, target]

            # Check if U_target is in span of U_others
            try:
                # Use least squares
                coeffs, residuals, rank_ls, _ = np.linalg.lstsq(U_others, U_target, rcond=1e-10)
            except:
                continue

            # Check residual
            U_recon = U_others @ coeffs
            if np.allclose(U_target, U_recon, atol=1e-8):
                # U_target = sum coeffs[i] * U[:, others[i]]
                # Now check if V_target and W_target can be adjusted
                # U[:,target]*V[:,target]*W[:,target] = sum_i coeffs[i]*U[:,others[i]]*V[:,target]*W[:,target]
                # We need: V[:,target]*W[:,target] = sum_i coeffs[i]*V[:,others[i]]*W[:,others[i]]
                # This is generally NOT satisfied, so we can't simply eliminate
                # Skip for now
                pass

        return U, V, W, eliminated

    def try_rank_reduction(U, V, W):
        """Apply all reduction strategies."""
        U, V, W, e1 = reduce_by_scalar_proportionality(U, V, W)
        U, V, W, e2 = reduce_by_linear_combination(U, V, W)
        U, V, W, e3 = reduce_by_column_dependence(U, V, W)
        return U, V, W, e1 + e2 + e3

    def count_nonzero_cols(U, V, W):
        rank = U.shape[1]
        nz = []
        for c in range(rank):
            if (np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0)):
                nz.append(c)
        return nz

    def to_clean_list(mat):
        rows = []
        for row in mat:
            clean_row = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded) < 1e-10:
                    rounded = 0.0
                clean_row.append(rounded)
            rows.append(clean_row)
        return rows

    # Try all builders and pick the best
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')

    builders = [build_strassen_hybrid, build_strassen_hybrid_v2, build_full_schoolbook]
    for builder in builders:
        if not _time_ok(40):
            break
        try:
            terms = builder()
            U_t, V_t, W_t = terms_to_matrices(terms)
            if not verify_full(U_t, V_t, W_t):
                continue

            # Try reduction
            U_r, V_r, W_r, _ = try_rank_reduction(U_t, V_t, W_t)

            if verify_full(U_r, V_r, W_r, tol=1e-8):
                nz = count_nonzero_cols(U_r, V_r, W_r)
                r = len(nz)
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r[:, nz], V_r[:, nz], W_r[:, nz]
            else:
                nz = count_nonzero_cols(U_t, V_t, W_t)
                r = len(nz)
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_t[:, nz], V_t[:, nz], W_t[:, nz]
        except:
            continue

    if best_U is None:
        terms = build_strassen_hybrid()
        best_U, best_V, best_W = terms_to_matrices(terms)

    U, V, W = best_U, best_V, best_W

    # Final cleanup
    U_list = to_clean_list(U)
    V_list = to_clean_list(V)
    W_list = to_clean_list(W)

    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)

    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)

    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)

    nz_cols = count_nonzero_cols(U_np, V_np, W_np)
    U_np = U_np[:, nz_cols]
    V_np = V_np[:, nz_cols]
    W_np = W_np[:, nz_cols]

    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)
    else:
        U_list = to_clean_list(U_np)
        V_list = to_clean_list(V_np)
        W_list = to_clean_list(W_np)

    return {"U": U_list, "V": V_list, "W": W_list}
