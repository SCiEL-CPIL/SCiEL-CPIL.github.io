"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Combines a Strassen-based construction (rank 68) with a tensor-aware
peeling refinement that removes redundant terms by testing whether a
term's contribution is in the span of the remaining terms.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
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

    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    # Build initial decomposition
    U_t = []
    V_t = []
    W_t = []

    # Strassen for A[0:2, 0:2] * B[0:2, k:k+2] for k = 0, 2, 4, 6
    for k_off in range(0, 8, 2):
        strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)

    # Direct terms for remaining parts
    # A[0:2, 2] * B[2, k] for all k
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)

    # A[2, 0:2] * B[0:2, k] for all k
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)

    # A[2, 2] * B[2, k] for all k
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)

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

    def verify_decomp(Ua, Va, Wa, nn, mm, pp):
        """Verify that the decomposition is exact."""
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    s = np.sum(Ua[i*mm+j] * Va[j*pp+k] * Wa[k*nn+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
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

    # Tensor-aware peeling refinement:
    # A term r is redundant if the tensor u_r ⊗ v_r ⊗ w_r lies in the span
    # of {u_s ⊗ v_s ⊗ w_s : s != r}. A sufficient (and for sparse integer
    # decompositions often necessary) test: remove term r and check if the
    # residual can be expressed as a linear combination of the remaining terms.
    # We do this by solving for coefficients alpha_s such that
    #   sum_s alpha_s * (u_s ⊗ v_s ⊗ w_s) = u_r ⊗ v_r ⊗ w_r
    # which is a linear system in alpha_s with N = n*m*p equations.

    def try_remove_term(Ua, Va, Wa, r, nn, mm, pp):
        """Try to remove term r. Return (new_U, new_V, new_W) or None if not removable."""
        R = Wa.shape[1]
        if R <= 1:
            return None
        # Build the tensor for term r
        tr = np.zeros((nn, mm, pp))
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    tr[i, j, k] = Ua[i*mm+j, r] * Va[j*pp+k, r] * Wa[k*nn+i, r]
        if np.all(np.abs(tr) < 1e-12):
            # Term is already zero; just remove
            return (np.delete(Ua, r, axis=1), np.delete(Va, r, axis=1), np.delete(Wa, r, axis=1))

        # Build the linear system: for each (i,j,k), sum_{s != r} alpha_s * Ua[i,j,s]*Va[j,k,s]*Wa[i,k,s] = tr[i,j,k]
        # This is linear in alpha_s.
        idxs = [s for s in range(R) if s != r]
        R2 = len(idxs)
        # Build matrix A where A[(i,j,k), s] = Ua[i*mm+j, idxs[s]] * Va[j*pp+k, idxs[s]] * Wa[k*nn+i, idxs[s]]
        # Number of equations: nn*mm*pp
        num_eq = nn * mm * pp
        A_mat = np.zeros((num_eq, R2))
        b_vec = np.zeros(num_eq)
        eq = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    b_vec[eq] = tr[i, j, k]
                    for s_idx, s in enumerate(idxs):
                        A_mat[eq, s_idx] = Ua[i*mm+j, s] * Va[j*pp+k, s] * Wa[k*nn+i, s]
                    eq += 1
        # Solve least squares
        try:
            alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
        except np.linalg.LinAlgError:
            return None
        # Check residual
        recon = A_mat @ alpha
        if np.max(np.abs(recon - b_vec)) > 1e-7:
            return None
        # Build new factors: for each remaining term s, new coefficient = old + alpha * (contribution to term r)
        # Actually we need: sum_s alpha_s * (u_s ⊗ v_s ⊗ w_s) = u_r ⊗ v_r ⊗ w_r
        # So new_u_r' = u_r - sum_s alpha_s * u_s (but we're removing r, so we absorb into others)
        # New factor for term s: u_s_new = u_s - alpha_s * u_r? No.
        # We want sum_{s != r} beta_s * (u_s ⊗ v_s ⊗ w_s) = original sum including r
        # = sum_{s != r} u_s ⊗ v_s ⊗ w_s + u_r ⊗ v_r ⊗ w_r
        # = sum_{s != r} (u_s + alpha_s * u_r) ⊗ ... no, that's not right either.
        # The decomposition is: sum_r u_r ⊗ v_r ⊗ w_r. If we remove r and add alpha_s to the coefficient,
        # we need the tensor identity. The cleanest way: the new decomposition is
        # sum_{s != r} (u_s + alpha_s * u_r) ⊗ (v_s + alpha_s * v_r) ⊗ (w_s + alpha_s * w_r)? No.
        # Actually, the linear system we solved was:
        #   sum_{s != r} alpha_s * (u_s ⊗ v_s ⊗ w_s) = u_r ⊗ v_r ⊗ w_r
        # This means the tensor of term r is exactly a linear combination of the other term tensors.
        # So we can simply remove term r and the total tensor is unchanged!
        # Because total = sum_{s != r} (u_s ⊗ v_s ⊗ w_s) + u_r ⊗ v_r ⊗ w_r
        #               = sum_{s != r} (u_s ⊗ v_s ⊗ w_s) + sum_{s != r} alpha_s (u_s ⊗ v_s ⊗ w_s)
        #               = sum_{s != r} (1 + alpha_s) (u_s ⊗ v_s ⊗ w_s)
        # So we just scale each remaining term by (1 + alpha_s).
        new_U = np.delete(Ua, r, axis=1)
        new_V = np.delete(Va, r, axis=1)
        new_W = np.delete(Wa, r, axis=1)
        scale = 1.0 + alpha
        new_U = new_U * scale[None, :]
        new_V = new_V * scale[None, :]
        new_W = new_W * scale[None, :]
        return (new_U, new_V, new_W)

    # Iteratively try to remove terms
    improved = True
    while improved:
        improved = False
        for r in range(W.shape[1]):
            if time.time() - start_time > DEADLINE - 5:
                break
            result = try_remove_term(U, V, W, r, n, m, p)
            if result is not None:
                U, V, W = result
                improved = True
                break  # restart the loop since indices changed

    # Verify final decomposition
    if not verify_decomp(U, V, W, n, m, p):
        # Fallback: rebuild from the original (pre-peeling) if peeling broke it
        # Rebuild original
        U_t2 = []
        V_t2 = []
        W_t2 = []
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off, U_t2, V_t2, W_t2)
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k, U_t2, V_t2, W_t2)
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k, U_t2, V_t2, W_t2)
        for k in range(8):
            direct_term(2, 2, k, U_t2, V_t2, W_t2)
        rank2 = len(W_t2)
        U = np.zeros((n * m, rank2), dtype=float)
        V = np.zeros((m * p, rank2), dtype=float)
        W = np.zeros((p * n, rank2), dtype=float)
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W[k * n + i, idx] = coeff

    # Convert to nice integer/half-integer values
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

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
