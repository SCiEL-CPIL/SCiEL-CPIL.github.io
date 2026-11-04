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
    # This gives rank 66.

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

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

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

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

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

    if not verify(U, V, W, 66):
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

    # Build the tensor flattening matrix for nullspace analysis
    # M[xyz, l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
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

    # Try to reduce rank by finding nullspace vectors
    best_rank = 66
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 1: Find nullspace vectors and eliminate columns
    M = build_tensor_matrix(U, V, W, 66)
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Check for small singular values
        null_dim = 0
        for idx in range(len(s) - 1, -1, -1):
            if s[idx] < 1e-8:
                null_dim += 1
            else:
                break
        
        if null_dim > 0:
            # For each nullspace vector, try to eliminate the column with largest coefficient
            for ns_idx in range(len(s) - null_dim, len(s)):
                if time.time() > _deadline - 30:
                    break
                null_vec = Vt[ns_idx, :]
                max_idx = np.argmax(np.abs(null_vec))
                
                # Try eliminating column max_idx
                U_try = np.delete(U, max_idx, axis=1)
                V_try = np.delete(V, max_idx, axis=1)
                W_try = np.delete(W, max_idx, axis=1)
                
                if verify(U_try, V_try, W_try, 65):
                    best_rank = 65
                    best_U, best_V, best_W = U_try, V_try, W_try
                    U, V, W = U_try, V_try, W_try
                    # Rebuild M for further reductions
                    M = build_tensor_matrix(U, V, W, 65)
                    try:
                        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                    except:
                        break
    except:
        pass

    # Phase 2: Greedy column elimination with residual analysis
    # Try removing each column and check if the remaining decomposition still works
    current_rank = len(U)
    for attempt in range(3):
        if time.time() > _deadline - 25:
            break
        if current_rank <= best_rank:
            break

        improved = False
        for remove_idx in range(current_rank):
            if time.time() > _deadline - 20:
                break

            U_rem = np.delete(U, remove_idx, axis=1)
            V_rem = np.delete(V, remove_idx, axis=1)
            W_rem = np.delete(W, remove_idx, axis=1)

            # Check if the remaining terms alone can reconstruct the tensor
            if verify(U_rem, V_rem, W_rem, current_rank - 1):
                best_rank = current_rank - 1
                best_U, best_V, best_W = U_rem, V_rem, W_rem
                U, V, W = U_rem, V_rem, W_rem
                current_rank = best_rank
                improved = True
                break

        if not improved:
            break

    if best_rank < 66:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
