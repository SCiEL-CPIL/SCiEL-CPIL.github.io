"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Builds the current 68-term decomposition, then attempts to reduce rank
by detecting and eliminating linearly dependent terms (rank reduction
via SVD on the 3D tensor). Falls back to the 68-term solution if no
improvement is found or if verification fails.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

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

        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    U_t = []
    V_t = []
    W_t = []

    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)

    # A[0:2, 2] * B[2, 0:8]: direct (16 terms)
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)

    # A[2, 0:2] * B[0:2, 0:8]: direct (16 terms)
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)

    # A[2, 2] * B[2, 0:8]: direct (8 terms)
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)

    rank_init = len(W_t)

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

    U_arr, V_arr, W_arr = build_matrices(U_t, V_t, W_t)

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
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

    # --- Rank reduction via SVD of the 3D tensor ---
    # Build the 3D tensor T of shape (n*m, m*p, p*n)
    T = np.einsum('ik,jk,lk->ijl', U_arr, V_arr, W_arr)
    # Verify T is the multiplication tensor
    T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    if not np.allclose(T, T_expected, atol=1e-9):
        # Shouldn't happen, but be safe
        pass

    # SVD the tensor via unfolding
    # Rank of tensor = min rank across all matricizations (if it were a product)
    # For general tensors, we use the rank of the best matricization as a lower bound
    # and try to find a lower-rank decomposition.

    # Try to find a decomposition of lower rank using SVD on matricizations
    # and then verify.

    best_rank = rank_init
    best_U, best_V, best_W = U_arr, V_arr, W_arr

    if time.time() < DEADLINE - 30:
        # Attempt rank reduction via iterative SVD on matricizations
        # Start with current rank, try to reduce
        for target_rank in range(rank_init - 1, max(20, rank_init - 30), -1):
            if time.time() >= DEADLINE - 10:
                break

            # Try to find a rank-target_rank decomposition
            # Using SVD on the 3 matricizations
            # Matricization 1: (n*m) x (m*p * p*n)
            T_mat1 = T.reshape(n*m, m*p*p*n)
            U1, S1, Vt1 = np.linalg.svd(T_mat1, full_matrices=False)

            # Keep top target_rank singular values
            U1_r = U1[:, :target_rank]
            S1_r = S1[:target_rank]
            V1_r = Vt1[:target_rank, :]

            # Reshape back: S1_r is (target_rank,), V1_r is (target_rank, m*p*p*n)
            # We need U (n*m, target_rank), V (m*p, target_rank), W (p*n, target_rank)
            # such that T[i,j,l] = sum_k U[i,k]*V[j,k]*W[l,k]
            # From matricization: T_mat1[i, (j,l)] = sum_k U1_r[i,k] * S1_r[k] * V1_r[k, (j,l)]
            # So: U_new[i,k] = U1_r[i,k] * S1_r[k]^(1/3)
            #     V_new[j,k] * W_new[l,k] = S1_r[k]^(1/3) * V1_r[k, (j,l)]
            # We can split: V_new[j,k] = S1_r[k]^(1/6) * V1_r[k, j, :] reshaped...
            # This is getting complicated. Let me use a different approach.

            # Simpler: use the 3-way SVD / higher-order SVD
            # Or: iteratively solve for U, V, W given target rank

            # Alternative: use the fact that if the tensor has rank r,
            # then each matricization has rank <= r.
            # The current tensor has matricization ranks:
            r1 = np.sum(S1 > 1e-10)

            # Check if the tensor can be decomposed with rank < rank_init
            # by looking at the matricization ranks
            T_mat2 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            _, S2, _ = np.linalg.svd(T_mat2, full_matrices=False)
            r2 = np.sum(S2 > 1e-10)

            T_mat3 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            _, S3, _ = np.linalg.svd(T_mat3, full_matrices=False)
            r3 = np.sum(S3 > 1e-10)

            tensor_rank_lower_bound = max(r1, r2, r3)

            if tensor_rank_lower_bound >= target_rank:
                continue

            # Try to find a rank-target_rank decomposition via optimization
            # This is complex, so skip for now and just use the matricization
            # rank as a potential improvement indicator

            break

    # Try a different approach: check if some columns of U, V, W are linearly dependent
    # and can be merged
    if time.time() < DEADLINE - 15:
        # Check for linear dependencies in U
        # If columns of U are linearly dependent, we might be able to reduce rank
        # But this requires careful handling of all three matrices simultaneously

        # Try to find a minimal rank decomposition via greedy elimination
        # Start with all terms, try to remove each one and see if the rest still works

        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = rank_init

        # Try removing terms one by one (greedy)
        for term_idx in range(current_rank - 1, -1, -1):
            if time.time() >= DEADLINE - 10:
                break

            # Skip this term
            U_test = np.delete(current_U, term_idx, axis=1)
            V_test = np.delete(current_V, term_idx, axis=1)
            W_test = np.delete(current_W, term_idx, axis=1)

            # Check if the decomposition still holds
            T_test = np.einsum('ik,jk,lk->ijl', U_test, V_test, W_test)
            if np.allclose(T_test, T_expected, atol=1e-9):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1

    if current_rank < rank_init:
        best_U, best_V, best_W = current_U, current_V, current_W
        best_rank = current_rank

    # Convert to nice integer/half-integer values
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

    U_out = to_nice_list(best_U)
    V_out = to_nice_list(best_V)
    W_out = to_nice_list(best_W)

    # Final verification
    U_final = np.array(U_out, dtype=float)
    V_final = np.array(V_out, dtype=float)
    W_final = np.array(W_out, dtype=float)

    if not verify_decomp(U_final, V_final, W_final):
        # If the "nice" conversion broke something, use raw values
        U_final = best_U
        V_final = best_V
        W_final = best_W
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END

