"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a hybrid approach: Strassen for 2x2 block of A against 2x8 block of B,
plus direct terms for the remaining row/col. Aggressive rank reduction via
iterative linear dependency detection on the tensor slice matrix.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0

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

    rank = len(W_t)

    # Build factor matrices
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

    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

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

    # Aggressive iterative rank reduction
    # Build the tensor matrix: each column r is the vectorized outer product
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # If column r is in the span of other columns, we can remove it.
    # But removal must preserve the exact decomposition, so we need
    # to check if the tensor slice (as a 3D array) can be expressed as
    # a linear combination of other slices.

    def build_tensor_mat(Ua, Va, Wa, rk):
        tm = np.zeros((n*m*p, rk), dtype=float)
        for r in range(rk):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tm[i*m*p + j*p + k, r] = Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
        return tm

    def try_remove_columns(Ua, Va, Wa, rk, max_iter=20):
        """Iteratively try to remove columns that are linear combinations of others."""
        tm = build_tensor_mat(Ua, Va, Wa, rk)
        removed = set()
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            found = False
            for r in range(rk - 1, -1, -1):
                if r in removed:
                    continue
                if time.time() - start_time > DEADLINE:
                    break
                # Get all other non-removed columns
                others = [c for c in range(rk) if c != r and c not in removed]
                if not others:
                    continue
                others_mat = tm[:, others]
                target = tm[:, r]
                try:
                    coeffs, residuals, rk_lstsq, sv = np.linalg.lstsq(others_mat, target, rcond=1e-12)
                    recon = others_mat @ coeffs
                    if np.allclose(recon, target, atol=1e-10, rtol=1e-10):
                        removed.add(r)
                        found = True
                        break
                except:
                    pass
            if not found:
                break
        return removed

    removed = try_remove_columns(U, V, W, rank, max_iter=30)

    if removed:
        keep = [i for i in range(rank) if i not in removed]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
        if verify_decomp(U, V, W):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank

    # Second pass: try removing pairs of columns that are linearly dependent
    # (i.e., their sum or difference is in the span)
    def try_remove_pairs(Ua, Va, Wa, rk, max_iter=10):
        tm = build_tensor_mat(Ua, Va, Wa, rk)
        removed = set()
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            found = False
            for r1 in range(rk - 1, -1, -1):
                if r1 in removed:
                    continue
                for r2 in range(r1 - 1, -1, -1):
                    if r2 in removed:
                        continue
                    if time.time() - start_time > DEADLINE:
                        break
                    # Check if column r1 is in span of others (excluding r2)
                    others = [c for c in range(rk) if c != r1 and c != r2 and c not in removed]
                    if not others:
                        continue
                    others_mat = tm[:, others]
                    target = tm[:, r1]
                    try:
                        coeffs, residuals, rk_lstsq, sv = np.linalg.lstsq(others_mat, target, rcond=1e-12)
                        recon = others_mat @ coeffs
                        if np.allclose(recon, target, atol=1e-10, rtol=1e-10):
                            removed.add(r1)
                            found = True
                            break
                    except:
                        pass
                if found:
                    break
            if not found:
                break
        return removed

    if time.time() - start_time < DEADLINE - 10:
        removed2 = try_remove_pairs(U, V, W, rank, max_iter=15)
        if removed2:
            keep = [i for i in range(rank) if i not in removed2]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            if verify_decomp(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = len(keep)
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank

    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Rebuild from scratch
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off)
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        for k in range(8):
            direct_term(2, 2, k)
        rank2 = len(W_t2)
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U2[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V2[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W2[k * n + i, idx] = coeff
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    def to_nice_list(arr):
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

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
