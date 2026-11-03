"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6
    nmp = n * m * p  # 72

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_strassen_66():
        """Build the rank-66 Strassen-based decomposition."""
        U = [[0.0] * 66 for _ in range(n * m)]
        V = [[0.0] * 66 for _ in range(m * p)]
        W = [[0.0] * 66 for _ in range(p * n)]
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

            # M1 = (A00+A11)*(B00+B11)
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            # M2 = (A10+A11)*B00
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            # M3 = A00*(B01-B11)
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            # M4 = A11*(B10-B00)
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            # M5 = (A00+A01)*B11
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            # M6 = (A10-A00)*(B00+B01)
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            # M7 = (A01-A11)*(B10+B11)
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
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

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        # Row 0: 6 blocks of 1x2*2x2
        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)
        # Rows 1-2: 6 blocks of 2x2*2x2 (Strassen)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        assert col == 66
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    def build_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_eliminate(U, V, W, r, max_eliminations=3):
        """Try to eliminate columns via null-space search."""
        best = (U.copy(), V.copy(), W.copy(), r)
        for elim_round in range(max_eliminations):
            if time.time() > _deadline - 10:
                break
            cur_r = best[3]
            M = np.zeros((nmp, cur_r))
            for l in range(cur_r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[i * m * p + j * p + k, l] = \
                                best[0][i * m + j, l] * best[1][j * p + k, l] * best[2][k * n + i, l]
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = int(np.sum(s > tol))
                if rank_M >= cur_r:
                    break
                null_space = Vt[rank_M:, :].T  # (cur_r, cur_r - rank_M)
                eliminated = False
                for nv in null_space:
                    if time.time() > _deadline - 10:
                        break
                    max_idx = int(np.argmax(np.abs(nv)))
                    if abs(nv[max_idx]) < tol:
                        continue
                    U_new = np.delete(best[0], max_idx, axis=1)
                    V_new = np.delete(best[1], max_idx, axis=1)
                    W_new = np.delete(best[2], max_idx, axis=1)
                    if verify(U_new, V_new, W_new, cur_r - 1):
                        best = (U_new, V_new, W_new, cur_r - 1)
                        eliminated = True
                        break
                if not eliminated:
                    break
            except:
                break
        return best

    # Build rank-66
    U66, V66, W66 = build_strassen_66()
    if not verify(U66, V66, W66, 66):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to eliminate
    U_best, V_best, W_best, r_best = try_eliminate(U66, V66, W66, 66)

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
