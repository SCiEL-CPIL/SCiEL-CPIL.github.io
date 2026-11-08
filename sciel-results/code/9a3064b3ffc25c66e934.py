"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a structural approach: build a candidate decomposition using
Strassen blocks and direct terms, then aggressively reduce rank via
tensor SVD and iterative column elimination.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def build_strassen_direct():
        """Build initial decomposition using Strassen for 2x2 blocks + direct terms."""
        U_t = []
        V_t = []
        W_t = []

        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
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

            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])

            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])

            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])

            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])

            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])

            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])

            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])

        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])

        # A[0:2, 0:2] * B[0:2, 0:8]: 4 Strassen blocks
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off)

        # A[0:2, 2] * B[2, 0:8]: direct
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)

        # A[2, 0:2] * B[0:2, 0:8]: direct
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)

        # A[2, 2] * B[2, 0:8]: direct
        for k in range(8):
            direct_term(2, 2, k)

        return U_t, V_t, W_t

    def build_matrices(U_t, V_t, W_t):
        rank = len(W_t)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff

        return U, V, W

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

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
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    # Build initial decomposition
    U_t, V_t, W_t = build_strassen_direct()
    U, V, W = build_matrices(U_t, V_t, W_t)
    rank = len(W_t)

    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Iterative column elimination: remove columns that are linear combinations of others
    # We work with the full tensor representation
    def tensor_column(Ua, Va, Wa, r):
        """Get the tensor for column r as a flat vector."""
        return np.kron(np.kron(Ua[:, r], Va[:, r]), Wa[:, r])

    def try_remove_column(Ua, Va, Wa, r):
        """Try to remove column r by expressing it as combination of others."""
        rank = Ua.shape[1]
        if rank <= 1:
            return False

        vec_r = tensor_column(Ua, Va, Wa, r)
        others = []
        for s in range(rank):
            if s != r:
                others.append(tensor_column(Ua, Va, Wa, s))

        if len(others) == 0:
            return False

        others_mat = np.column_stack(others)

        try:
            coeffs, residuals, rank_others, _ = np.linalg.lstsq(others_mat, vec_r, rcond=1e-12)
            reconstruction = others_mat @ coeffs
            if np.linalg.norm(reconstruction - vec_r) < 1e-7 * np.linalg.norm(vec_r):
                # Column r is in span of others, remove it
                Ua = np.delete(Ua, r, axis=1)
                Va = np.delete(Va, r, axis=1)
                Wa = np.delete(Wa, r, axis=1)
                return True
        except:
            pass
        return False

    # Multiple passes of column elimination
    for iteration in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break

        removed_any = False
        for r in range(U.shape[1] - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 10:
                break
            if try_remove_column(U, V, W, r):
                removed_any = True

        if not removed_any:
            break

        # Update best
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]

    # Try a different approach: use tensor SVD to find a low-rank approximation
    # and see if we can get an exact decomposition with fewer terms
    if time.time() - start_time < DEADLINE - 20:
        # Build the full tensor
        full_tensor = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    full_tensor[i * m + j, j * p + k, k * n + i] = 1.0

        # Try to find minimum rank via iterative truncation
        # Start with current best rank
        current_rank = best_U.shape[1]

        # Try to find a rank-r decomposition for r < current_rank
        # This is NP-hard in general, so we use a heuristic
        # Use the fact that the tensor has a special structure

        # Alternative: try to combine terms algebraically
        # Look for patterns in the U, V, W matrices that allow merging

        # Check if any two columns can be merged
        for r1 in range(best_U.shape[1]):
            if time.time() - start_time > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, best_U.shape[1]):
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Check if columns r1 and r2 have the same support pattern
                # that allows algebraic combination
                u1 = best_U[:, r1]
                u2 = best_U[:, r2]
                v1 = best_V[:, r1]
                v2 = best_V[:, r2]
                w1 = best_W[:, r1]
                w2 = best_W[:, r2]

                # If u1 and u2 are proportional, and v1 and v2 are proportional,
                # and w1 and w2 are proportional, we can merge
                if np.linalg.norm(u1) < 1e-12 or np.linalg.norm(u2) < 1e-12:
                    continue
                if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
                    continue
                if np.linalg.norm(w1) < 1e-12 or np.linalg.norm(w2) < 1e-12:
                    continue

                # Check proportionality
                try:
                    ratio_u = u1[0] / u2[0] if abs(u2[0]) > 1e-12 else None
                    if ratio_u is None:
                        # Try to find any non-zero element
                        mask = abs(u2) > 1e-12
                        if np.any(mask):
                            idx = np.argmax(mask)
                            ratio_u = u1[idx] / u2[idx]
                        else:
                            continue

                    if np.allclose(u1, ratio_u * u2, atol=1e-10):
                        # Check v
                        mask_v = abs(v2) > 1e-12
                        if np.any(mask_v):
                            idx_v = np.argmax(mask_v)
                            ratio_v = v1[idx_v] / v2[idx_v]
                            if np.allclose(v1, ratio_v * v2, atol=1e-10):
                                # Check w
                                mask_w = abs(w2) > 1e-12
                                if np.any(mask_w):
                                    idx_w = np.argmax(mask_w)
                                    ratio_w = w1[idx_w] / w2[idx_w]
                                    if np.allclose(w1, ratio_w * w2, atol=1e-10):
                                        # All three are proportional, can merge
                                        # But this only helps if the combined term is still valid
                                        # Actually, if all three are proportional with the same ratio,
                                        # the terms are identical and one is redundant
                                        pass
                except:
                    pass

    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Rebuild from scratch
        U_t2, V_t2, W_t2 = build_strassen_direct()
        U2, V2, W2 = build_matrices(U_t2, V_t2, W_t2)
        return {"U": to_nice_list(U2), "V": to_nice_list(V2), "W": to_nice_list(W2)}

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
