"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build a block-structured decomposition:
    # - Split columns into 3 groups of 2
    # - For each group, use Strassen for the 2x2 sub-blocks in rows 1-2
    # - For row 0, use direct terms
    # This gives rank 66. Then attempt to reduce rank by finding
    # linear dependences among the rank-1 terms.

    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        # Direct: A0*B00 -> C0, A1*B10 -> C0
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        # Direct: A0*B01 -> C1, A1*B11 -> C1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    # Build the decomposition
    # Row 0: direct terms for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_tensor_matrix(U_mat, V_mat, W_mat, r):
        M = np.zeros((n * m * p, r))
        for l in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, l] = (
                            U_mat[i * m + j, l] * V_mat[j * p + k, l] * W_mat[k * n + i, l]
                        )
        return M

    # Phase 1: SVD-based nullspace detection for single-column elimination
    M = build_tensor_matrix(U, V, W, 66)
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        null_dim = 0
        for idx in range(len(s) - 1, -1, -1):
            if s[idx] < 1e-8:
                null_dim += 1
            else:
                break

        if null_dim > 0:
            for ns_idx in range(len(s) - null_dim, len(s)):
                if time.time() > _deadline - 40:
                    break
                null_vec = Vt[ns_idx, :]
                max_idx = int(np.argmax(np.abs(null_vec)))

                U_try = np.delete(U, max_idx, axis=1)
                V_try = np.delete(V, max_idx, axis=1)
                W_try = np.delete(W, max_idx, axis=1)

                if verify(U_try, V_try, W_try, len(U_try)):
                    U, V, W = U_try, V_try, W_try
                    M = build_tensor_matrix(U, V, W, len(U))
                    try:
                        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                    except:
                        break
    except:
        pass

    # Phase 2: Greedy single-column elimination with full verification
    current_rank = len(U)
    for attempt in range(5):
        if time.time() > _deadline - 30:
            break
        improved = False
        for remove_idx in range(current_rank):
            if time.time() > _deadline - 25:
                break
            U_rem = np.delete(U, remove_idx, axis=1)
            V_rem = np.delete(V, remove_idx, axis=1)
            W_rem = np.delete(W, remove_idx, axis=1)

            if verify(U_rem, V_rem, W_rem, current_rank - 1):
                U, V, W = U_rem, V_rem, W_rem
                current_rank -= 1
                improved = True
                break

        if not improved:
            break

    # Phase 3: Try small perturbations to find alternative decompositions
    # that might allow further reduction
    if time.time() < _deadline - 50:
        # Try adding a small random perturbation and re-optimizing
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = len(U)

        for trial in range(3):
            if time.time() > _deadline - 45:
                break
            # Create a perturbed version
            eps = 1e-6 * (trial + 1)
            U_pert = U + eps * np.random.randn(*U.shape)
            V_pert = V + eps * np.random.randn(*V.shape)
            W_pert = W + eps * np.random.randn(*W.shape)

            # Try to reduce the perturbed decomposition
            for remove_idx in range(len(U_pert)):
                if time.time() > _deadline - 40:
                    break
                U_rem = np.delete(U_pert, remove_idx, axis=1)
                V_rem = np.delete(V_pert, remove_idx, axis=1)
                W_rem = np.delete(W_pert, remove_idx, axis=1)

                if verify(U_rem, V_rem, W_rem, len(U_rem)):
                    if len(U_rem) < best_rank:
                        best_rank = len(U_rem)
                        best_U, best_V, best_W = U_rem.copy(), V_rem.copy(), W_rem.copy()
                    break

        if best_rank < len(U):
            U, V, W = best_U, best_V, best_W

    # Final verification and return
    if not verify(U, V, W, len(U)):
        # Fallback to the original rank-66
        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
