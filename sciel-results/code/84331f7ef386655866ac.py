def construct():
    import time
    import numpy as np
    _t0 = time.time()
    _DEADLINE = 145.0

    n, m, p = 3, 4, 6

    def build_strassen_hybrid():
        """Hybrid: schoolbook for row 0, Strassen 2x2x2 for rows 1-2 with column pairs."""
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
        
        # Schoolbook for row 0
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        # Strassen for rows 1,2 with column pairs (0,1), (2,3), (4,5)
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        return terms

    def build_strassen_hybrid_v2():
        """Hybrid: Strassen for rows 0-1, schoolbook for row 2."""
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
        
        # Strassen for rows 0,1
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(0, 1, 0, 1, k1, k2)
            add_strassen_block(0, 1, 2, 3, k1, k2)
        
        # Schoolbook for row 2
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(2, j, k)
        
        return terms

    def build_strassen_hybrid_v3():
        """Hybrid: Strassen for rows 0-2 with column pairs (0,1), (2,3), (4,5) using 3x3 Strassen-like."""
        terms = []
        
        # For 3 rows x 2 cols x 2 cols, we can use a 7-term decomposition similar to Strassen
        # But let's try a simpler approach: pair rows (0,1) with Strassen, then handle row 2
        # Or try all pairs: (0,1), (0,2), (1,2) and combine
        
        # Let's use the standard 2x2 Strassen for each pair of rows, covering all column pairs
        # This gives 3 pairs * 3 col_pairs * 7 = 63 terms, which is worse than 66
        # So stick with the v1 approach but try different row pairings
        
        # Actually, let's try: Strassen for (0,1) on cols (0,1), (2,3), (4,5) = 21 terms
        # Strassen for (0,2) on cols (0,1), (2,3), (4,5) = 21 terms  
        # Strassen for (1,2) on cols (0,1), (2,3), (4,5) = 21 terms
        # Total = 63, but we have overlaps. Let's just use v1 which gives 66.
        
        # Alternative: Use 3x2x2 blocks with a custom decomposition
        # For a 3x2x2 matrix multiplication, the minimum rank is 7 (same as 2x2x2)
        # But we need to cover 3 rows. Let's use:
        # - 2x2 Strassen for rows (0,1) on each col pair: 3*7 = 21
        # - 1x2x2 schoolbook for row 2 on each col pair: 3*4 = 12
        # Total = 33 for the Strassen part + 12 = 45? No, that's not right.
        
        # Let me just return the same as v1 for now, but with a different structure
        # Actually, let's try: schoolbook for row 0 (24 terms), Strassen for rows 1,2 (42 terms) = 66
        # This is the same as v1.
        
        # Let's try a completely different approach: use 2x2x4 Strassen-like
        # For 2 rows x 2 cols x 4 cols, we can potentially do better than 2*7=14
        # But this is complex. Let's just return v1's result for now.
        return build_strassen_hybrid()

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
        """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all valid (i,j,k)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_random(U, V, W, n_samples=500, tol=1e-9):
        rng = np.random.RandomState(12345)
        for _ in range(n_samples):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(val - 1.0) > tol:
                return False
        return True

    def reduce_rank_v2(U, V, W):
        """
        Advanced rank reduction:
        1. Find scalar-proportional column pairs and eliminate.
        2. Use linear algebra to find and eliminate linearly dependent columns.
        3. Try to find zero-sum combinations of terms.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        active = list(range(rank))
        eliminated = 0

        # Phase 1: Scalar proportionality elimination
        for _ in range(50):
            if time.time() - _t0 > _DEADLINE - 15:
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
                    ub = U[:, rb]
                    if np.allclose(ub, 0, atol=1e-12):
                        continue
                    nz_ub = np.nonzero(ub)[0]
                    if len(nz_ub) == 0:
                        continue
                    pos = nz_ub[0]
                    if abs(ub[pos]) < 1e-12:
                        continue
                    c = ua[pos] / ub[pos]
                    if abs(c) < 1e-12:
                        continue
                    if not np.allclose(ua, c * ub, atol=1e-10):
                        continue
                    va, vb = V[:, ra], V[:, rb]
                    nz_vb = np.nonzero(vb)[0]
                    if len(nz_vb) == 0:
                        continue
                    posv = nz_vb[0]
                    if abs(vb[posv]) < 1e-12:
                        continue
                    d = va[posv] / vb[posv]
                    if abs(d) < 1e-12:
                        continue
                    if not np.allclose(va, d * vb, atol=1e-10):
                        continue
                    e_needed = 1.0 / (c * d)
                    wa, wb = W[:, ra], W[:, rb]
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

        # Phase 2: Linear dependency elimination via SVD on U
        if time.time() - _t0 > _DEADLINE - 20:
            return U, V, W, eliminated

        active = [c for c in range(rank) if not np.allclose(U[:, c], 0, atol=1e-12)]
        if len(active) < 3:
            return U, V, W, eliminated

        U_active = U[:, active]
        try:
            U_s, U_svt = np.linalg.svd(U_active, full_matrices=False)
        except:
            return U, V, W, eliminated

        tol_s = 1e-8 * max(U_svt.shape[0], 1)
        num_pivot = sum(1 for s in U_svt if s > tol_s)

        if num_pivot < len(active):
            # Find null space vectors
            null_vecs = U_svt[num_pivot:]
            for nv in null_vecs:
                if time.time() - _t0 > _DEADLINE - 20:
                    break
                # nv[i] * U[:, active[i]] sums to ~0
                # Try to eliminate the column with largest coefficient
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) < 1e-10:
                    continue
                max_col = active[max_idx]
                coeff = nv[max_idx]

                # We need: U[:,max_col] = -(1/coeff) * sum_{i != max_idx} nv[i] * U[:,active[i]]
                # For the tensor: contribution of max_col is U[:,max_col]*V[:,max_col]*W[:,max_col]
                # = -(1/coeff) * sum_i nv[i] * U[:,active[i]] * V[:,max_col] * W[:,max_col]
                # We want to absorb this into existing terms.
                # This requires: for each i, V[:,active[i]]*W[:,active[i]] to be adjusted
                # which is generally not possible without changing the tensor.
                # Skip - too complex for now.
                pass

        return U, V, W, eliminated

    def reduce_rank_perturbation(U, V, W):
        """
        Try small perturbations to find rank reductions.
        If we can find coefficients c_r such that sum_r c_r * U[:,r]*V[:,r]*W[:,r] = 0,
        then we can use this to eliminate one term.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        eliminated = 0

        # Try to find a linear combination of terms that gives zero tensor
        # This is equivalent to finding a vector c such that
        # sum_r c_r * outer(U[:,r], V[:,r], W[:,r]) = 0 in tensor space
        # Flatten: tensor has n*m*p = 72 entries
        # Build matrix M where M[i,j,k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Then find null space of M

        tensor_size = n * m * p
        M = np.zeros((tensor_size, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    idx += 1

        # Find null space of M
        try:
            M_s, M_sv, M_vt = np.linalg.svd(M, full_matrices=False)
        except:
            return U, V, W, eliminated

        tol_m = 1e-8 * max(M_s.shape[0], 1)
        num_pivot_m = sum(1 for s in M_s if s > tol_m)

        if num_pivot_m < rank:
            null_vecs = M_vt[num_pivot_m:]
            for nv in null_vecs:
                if time.time() - _t0 > _DEADLINE - 25:
                    break
                # nv is in null space: sum_r nv[r] * M[:,r] = 0
                # This means sum_r nv[r] * U[:,r]*V[:,r]*W[:,r] = 0 (as tensor)
                # We can use this to eliminate one term.
                # Pick the term with largest |nv[r]| to eliminate
                max_r = np.argmax(np.abs(nv))
                if abs(nv[max_r]) < 1e-10:
                    continue
                # We want to express term max_r as combination of others
                # U[:,max_r]*V[:,max_r]*W[:,max_r] = -(1/nv[max_r]) * sum_{r != max_r} nv[r] * U[:,r]*V[:,r]*W[:,r]
                # This doesn't directly give us a way to eliminate unless the U vectors are proportional
                # For now, just check if this helps
                pass

        return U, V, W, eliminated

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
        if time.time() - _t0 > _DEADLINE - 40:
            break
        try:
            terms = builder()
            U_t, V_t, W_t = terms_to_matrices(terms)
            if not verify_full(U_t, V_t, W_t):
                continue
            # Try reduction
            U_r, V_r, W_r, _ = reduce_rank_v2(U_t, V_t, W_t)
            U_r, V_r, W_r, _ = reduce_rank_perturbation(U_r, V_r, W_r)
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

    # Remove zero columns
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
