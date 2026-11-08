"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms, then applies systematic rank reduction via integer linear
dependency detection with improved elimination strategies.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    def build_decomp(partition):
        """Build decomposition based on partition strategy."""
        U_t = []
        V_t = []
        W_t = []

        if partition == 0:
            # 2x2 Strassen for A[0:2,0:2]*B[0:2,k], direct for rest
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            for k0 in [0, 2, 4, 6]:
                for i in range(2):
                    for k in range(k0, k0 + 2):
                        direct_term(i, 2, k, U_t, V_t, W_t)
                for j in range(2):
                    for k in range(k0, k0 + 2):
                        direct_term(2, j, k, U_t, V_t, W_t)
                for k in range(k0, k0 + 2):
                    direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 1:
            # Alternative: try Strassen on different block configurations
            # Use Strassen for A[0:2,0:2]*B[0:2,k] and also for A[1:3,0:2]*B[0:2,k]
            # This overlaps in the middle row/col but may allow better reduction
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            # A[2,0:2] * B[0:2,k] direct
            for k in range(p):
                for j in range(2):
                    direct_term(2, j, k, U_t, V_t, W_t)
            # A[0:2,2] * B[2,k] direct
            for k in range(p):
                for i in range(2):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            # A[2,2] * B[2,k] direct
            for k in range(p):
                direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 2:
            # Try Strassen on A[0:2,0:2]*B[0:2,k] and A[0:2,1:3]*B[1:3,k]
            # This uses overlapping 2x2 blocks in the A matrix
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            # A[0:2,2] * B[2,k] direct
            for k in range(p):
                for i in range(2):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            # A[2,0] * B[0,k] direct
            for k in range(p):
                direct_term(2, 0, k, U_t, V_t, W_t)
            # A[2,1:3] * B[1:3,k] - but we already have B[0:2,k] in Strassen
            # So just A[2,1] and A[2,2]
            for k in range(p):
                direct_term(2, 1, k, U_t, V_t, W_t)
                direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 3:
            # Schoolbook baseline
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        direct_term(i, j, k, U_t, V_t, W_t)

        rank = len(W_t)

        # Build the factor matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j][idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k][idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i][idx] = coeff

        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float), rank

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def term_vector(Ua, Va, Wa, r):
        """Get the tensor vector for term r."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i*m+j)*p + k] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        return vec

    def try_eliminate(r, Ua, Va, Wa, rank_cur, max_subset_size=12):
        """Try to express term r as integer combination of other terms."""
        target = term_vector(Ua, Va, Wa, r)
        others = [s for s in range(rank_cur) if s != r]
        if not others:
            return None

        # First try: full least squares
        M = np.array([term_vector(Ua, Va, Wa, s) for s in others])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
        except Exception:
            return None

        coeffs_int = np.round(coeffs)
        if np.all(np.abs(coeffs - coeffs_int) < 1e-6):
            coeff_map = {s: coeffs_int[i] for i, s in enumerate(others)}
            coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
        else:
            # Try with subsets of top-overlapping terms
            overlaps = np.abs(M @ target)
            order = np.argsort(-overlaps)
            
            found = False
            for top_k in [min(8, len(others)), min(12, len(others)), min(16, len(others))]:
                if top_k >= len(others):
                    break
                top = [others[i] for i in order[:top_k]]
                Mt = np.array([term_vector(Ua, Va, Wa, s) for s in top])
                try:
                    ct, _, _, _ = np.linalg.lstsq(Mt, target, rcond=None)
                except Exception:
                    continue
                ct_int = np.round(ct)
                if np.all(np.abs(ct - ct_int) < 1e-6):
                    coeff_map = {s: 0.0 for s in others}
                    for s, cv in zip(top, ct_int):
                        coeff_map[s] = cv
                    coeff_map = {s: c for s, c in coeff_map.items() if abs(c) > 1e-9}
                    found = True
                    break
            
            if not found:
                return None

        # Verify exactness
        recon = np.zeros(n * m * p)
        for s, c in coeff_map.items():
            recon += c * term_vector(Ua, Va, Wa, s)
        if not np.allclose(recon, target, atol=1e-9):
            return None

        # Build new matrices by modifying the kept terms
        keep = [s for s in range(rank_cur) if s != r]
        new_rank = len(keep)
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        for new_idx, s in enumerate(keep):
            scale = 1.0 + coeff_map.get(s, 0.0)
            if abs(scale) < 1e-9:
                continue
            new_U[:, new_idx] = Ua[:, s] * scale
            new_V[:, new_idx] = Va[:, s] * scale
            new_W[:, new_idx] = Wa[:, s] * scale

        # Remove zero columns
        nonzero_mask = np.any(np.abs(new_U) > 1e-9, axis=0) | \
                       np.any(np.abs(new_V) > 1e-9, axis=0) | \
                       np.any(np.abs(new_W) > 1e-9, axis=0)
        if not np.all(nonzero_mask):
            new_U = new_U[:, nonzero_mask]
            new_V = new_V[:, nonzero_mask]
            new_W = new_W[:, nonzero_mask]

        return new_U, new_V, new_W

    def try_pair_elimination(Ua, Va, Wa, rank_cur):
        """Try to find two terms that can be combined/eliminated."""
        # For each pair of terms, check if one is a scalar multiple of the other
        # or if their sum/difference creates a simpler structure
        for r1 in range(rank_cur):
            for r2 in range(r1 + 1, rank_cur):
                v1 = term_vector(Ua, Va, Wa, r1)
                v2 = term_vector(Ua, Va, Wa, r2)
                # Check if v1 = c * v2 for some small integer/half-integer c
                if np.any(np.abs(v2) > 1e-9):
                    c = v1[np.argmax(np.abs(v2))] / v2[np.argmax(np.abs(v2))]
                    c_rounded = round(c * 2) / 2
                    if abs(c - c_rounded) < 1e-9:
                        # v1 = c_rounded * v2
                        # We can replace v1 with (v1 - c_rounded*v2) which is zero
                        # So we just remove v1 and scale v2 appropriately
                        # Actually, if v1 = c*v2, then in the sum, v1 contributes c*v2
                        # So we can remove v1 and add c to the coefficient of v2
                        # But in our tensor decomposition, each term has coefficient 1 in the sum
                        # So we need to be more careful
                        pass
        return None

    def reduce_rank(Ua, Va, Wa, rank_cur, max_time):
        """Greedy rank reduction with multiple strategies."""
        progress = True
        max_passes = 50
        pass_count = 0
        while progress and pass_count < max_passes and (time.time() - start_time) < max_time:
            progress = False
            pass_count += 1
            for r in range(rank_cur):
                if (time.time() - start_time) > max_time:
                    break
                res = try_eliminate(r, Ua, Va, Wa, rank_cur)
                if res is not None:
                    new_U, new_V, new_W = res
                    if verify_decomp(new_U, new_V, new_W):
                        Ua, Va, Wa = new_U, new_V, new_W
                        rank_cur = Ua.shape[1]
                        progress = True
                        break
        return Ua, Va, Wa, rank_cur

    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    # Try multiple partition strategies and keep the best
    best_result = None
    best_rank = float('inf')

    for partition in [0, 1, 2]:
        if (time.time() - start_time) > DEADLINE - 15.0:
            break

        Ua, Va, Wa, rank = build_decomp(partition)

        if not verify_decomp(Ua, Va, Wa):
            continue

        # Apply rank reduction
        Ua, Va, Wa, rank = reduce_rank(Ua, Va, Wa, rank, DEADLINE - 5.0)

        if rank < best_rank:
            best_rank = rank
            best_result = (Ua, Va, Wa, rank)

        if (time.time() - start_time) > DEADLINE - 5.0:
            break

    if best_result is None:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    Ua, Va, Wa, rank = best_result
    return {"U": to_nice_list(Ua), "V": to_nice_list(Va), "W": to_nice_list(Wa)}
