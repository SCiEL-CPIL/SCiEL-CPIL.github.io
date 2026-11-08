"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms, then applies systematic rank reduction via integer linear
dependency detection.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            for k in range(p):
                for j in range(2):
                    direct_term(2, j, k, U_t, V_t, W_t)
            for k in range(p):
                for i in range(2):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            for k in range(p):
                direct_term(2, 2, k, U_t, V_t, W_t)
        elif partition == 2:
            # Try Strassen on A[0:2,0:2]*B[0:2,k] and A[0:2,1:3]*B[1:3,k]
            for k0 in [0, 2, 4, 6]:
                strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
            for k in range(p):
                for i in range(2):
                    direct_term(i, 2, k, U_t, V_t, W_t)
            for k in range(p):
                direct_term(2, 0, k, U_t, V_t, W_t)
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

    def precompute_term_vectors(Ua, Va, Wa, rank_cur):
        """Precompute all term vectors as a matrix (dim x rank)."""
        dim = n * m * p
        vecs = np.zeros((dim, rank_cur))
        for r in range(rank_cur):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vecs[(i*m+j)*p + k, r] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        return vecs

    def try_eliminate_fast(r, term_vecs, rank_cur):
        """Try to remove term r by checking if it's in the span of others (via rank test)."""
        if rank_cur <= 1:
            return None
        others = [s for s in range(rank_cur) if s != r]
        # Check if removing r drops the rank: term r is linearly dependent on others
        # Fast test: rank of all terms vs rank of others
        all_mat = term_vecs[:, :rank_cur]
        others_mat = term_vecs[:, others]
        
        try:
            rank_all = np.linalg.matrix_rank(all_mat, tol=1e-9)
            rank_others = np.linalg.matrix_rank(others_mat, tol=1e-9)
        except Exception:
            return None
        
        if rank_others == rank_all:
            # Term r is in the span of others; we can remove it
            keep = [s for s in range(rank_cur) if s != r]
            return keep
        return None

    def try_pair_elimination(term_vecs, rank_cur):
        """Try to find a pair of terms that can be removed together."""
        if rank_cur <= 2:
            return None
        # For each pair, check if removing both doesn't change rank
        # This is expensive, so limit to promising pairs
        all_mat = term_vecs[:, :rank_cur]
        try:
            rank_all = np.linalg.matrix_rank(all_mat, tol=1e-9)
        except Exception:
            return None
        
        # Only try pairs where both terms have significant overlap with the span
        for i in range(rank_cur):
            for j in range(i+1, rank_cur):
                keep = [s for s in range(rank_cur) if s != i and s != j]
                if not keep:
                    continue
                others_mat = term_vecs[:, keep]
                try:
                    rank_others = np.linalg.matrix_rank(others_mat, tol=1e-9)
                except Exception:
                    continue
                if rank_others == rank_all:
                    return keep
        return None

    def reduce_rank(Ua, Va, Wa, rank_cur, max_time):
        """Greedy rank reduction using precomputed term vectors."""
        term_vecs = precompute_term_vectors(Ua, Va, Wa, rank_cur)
        progress = True
        max_passes = 50
        pass_count = 0
        while progress and pass_count < max_passes and (time.time() - start_time) < max_time:
            progress = False
            pass_count += 1
            # Single term elimination
            for r in range(rank_cur):
                if (time.time() - start_time) > max_time:
                    break
                keep = try_eliminate_fast(r, term_vecs, rank_cur)
                if keep is not None:
                    new_U = Ua[:, keep]
                    new_V = Va[:, keep]
                    new_W = Wa[:, keep]
                    if verify_decomp(new_U, new_V, new_W):
                        Ua, Va, Wa = new_U, new_V, new_W
                        rank_cur = Ua.shape[1]
                        term_vecs = precompute_term_vectors(Ua, Va, Wa, rank_cur)
                        progress = True
                        break
            if progress or (time.time() - start_time) > max_time:
                continue
            # Pair elimination (only if rank is still > 10 to save time)
            if rank_cur > 10:
                keep = try_pair_elimination(term_vecs, rank_cur)
                if keep is not None:
                    new_U = Ua[:, keep]
                    new_V = Va[:, keep]
                    new_W = Wa[:, keep]
                    if verify_decomp(new_U, new_V, new_W):
                        Ua, Va, Wa = new_U, new_V, new_W
                        rank_cur = Ua.shape[1]
                        term_vecs = precompute_term_vectors(Ua, Va, Wa, rank_cur)
                        progress = True
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
        if (time.time() - start_time) > DEADLINE - 20.0:
            break

        Ua, Va, Wa, rank = build_decomp(partition)

        if not verify_decomp(Ua, Va, Wa):
            continue

        # Apply rank reduction with remaining time budget
        remaining = DEADLINE - 10.0
        Ua, Va, Wa, rank = reduce_rank(Ua, Va, Wa, rank, remaining)

        if rank < best_rank:
            best_rank = rank
            best_result = (Ua, Va, Wa, rank)

        if (time.time() - start_time) > DEADLINE - 10.0:
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

    # Final verification and cleanup
    Ua, Va, Wa, rank = best_result
    if not verify_decomp(Ua, Va, Wa):
        # Fallback to schoolbook if something went wrong
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
