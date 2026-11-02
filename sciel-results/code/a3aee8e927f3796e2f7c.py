import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start = time.time()
    deadline = start + 118.0  # safety margin under the 150s kill

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    # ---- Build the rank-64 Strassen block decomposition ----
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(np.asarray(u, dtype=float))
        V_cols.append(np.asarray(v, dtype=float))
        W_cols.append(np.asarray(w, dtype=float))

    def strassen_2x2(ai, aj, bj, bk):
        # 7 terms for multiplying a 2x2 (rows ai,ai+1; cols aj,aj+1)
        # by a 2x2 (rows bj,bj+1; cols bk,bk+1)
        # M1 = (a+d)(e+h)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai + 1, aj + 1)] = 1
        v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj + 1, bk + 1)] = 1
        w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk + 1, ai + 1)] = 1
        add_term(u, v, w)
        # M2 = (c+d)e
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai + 1, aj)] = 1; u[get_u_idx(ai + 1, aj + 1)] = 1
        v[get_v_idx(bj, bk)] = 1
        w[get_w_idx(bk, ai + 1)] = 1; w[get_w_idx(bk + 1, ai + 1)] = -1
        add_term(u, v, w)
        # M3 = a(f-h)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj)] = 1
        v[get_v_idx(bj, bk + 1)] = 1; v[get_v_idx(bj + 1, bk + 1)] = -1
        w[get_w_idx(bk + 1, ai)] = 1; w[get_w_idx(bk + 1, ai + 1)] = 1
        add_term(u, v, w)
        # M4 = d(g-e)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai + 1, aj + 1)] = 1
        v[get_v_idx(bj + 1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk + 1, ai)] = 1
        add_term(u, v, w)
        # M5 = (a+b)h
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj + 1)] = 1
        v[get_v_idx(bj + 1, bk + 1)] = 1
        w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk + 1, ai)] = 1
        add_term(u, v, w)
        # M6 = (c-a)(e+f)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai + 1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk + 1)] = 1
        w[get_w_idx(bk + 1, ai + 1)] = 1
        add_term(u, v, w)
        # M7 = (b-d)(g+h)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj + 1)] = 1; u[get_u_idx(ai + 1, aj + 1)] = -1
        v[get_v_idx(bj + 1, bk)] = 1; v[get_v_idx(bj + 1, bk + 1)] = 1
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def schoolbook_2x2x1(ai, aj, bj, bk):
        # 2 terms for multiplying a 2x2 (rows ai,ai+1; cols aj,aj+1)
        # by a 2x1 (rows bj,bj+1; col bk)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj)] = 1
        v[get_v_idx(bj, bk)] = 1
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai, aj + 1)] = 1
        v[get_v_idx(bj + 1, bk)] = 1
        w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai + 1, aj)] = 1
        v[get_v_idx(bj, bk)] = 1
        w[get_w_idx(bk, ai + 1)] = 1
        add_term(u, v, w)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[get_u_idx(ai + 1, aj + 1)] = 1
        v[get_v_idx(bj + 1, bk)] = 1
        w[get_w_idx(bk, ai + 1)] = 1
        add_term(u, v, w)

    # 8 blocks of (2x2 A) x (2x2 B) -> 2x2 output, using Strassen
    for ai in (0, 2):
        for bj in (0, 2):
            for bj2 in (0, 2):  # column block of B
                pass
    # A is 4x4: row blocks ai in {0,2}, col blocks aj in {0,2}
    # B is 4x5: row blocks bj in {0,2}, col blocks bk in {0,2,4}
    # output C is 4x5: row blocks ai, col blocks bk
    for ai in (0, 2):
        for aj in (0, 2):
            for bj in (0, 2):
                for bk in (0, 2):
                    strassen_2x2(ai, aj, bj, bk)  # 7 terms
            # column block bk=4 (the 5th column) is 2x1
            for bj in (0, 2):
                schoolbook_2x2x1(ai, aj, bj, 4)  # 4 terms

    U_mat = np.array(U_cols).T  # (16, rank)
    V_mat = np.array(V_cols).T  # (20, rank)
    W_mat = np.array(W_cols).T  # (20, rank)
    base_rank = U_mat.shape[1]

    # ---- Verification ----
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.integers(-3, 4, (n, m)).astype(float)
            B = rng.integers(-3, 4, (m, p)).astype(float)
            C_expected = A @ B
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j] @ (V[j * p + k] * W[k * n + i])
                    C_recon[i, k] = s
            if not np.allclose(C_recon, C_expected, atol=1e-6):
                return False
        return True

    best_rank = base_rank
    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()

    # ---- ALS refinement with half-integer rounding ----
    def round_half(x):
        return np.round(x * 2.0) / 2.0

    def als_round(U, V, W, iters):
        n, m, p = DIMENSIONS
        U = U.copy(); V = V.copy(); W = W.copy()
        rank = U.shape[1]
        for it in range(iters):
            if time.time() > deadline:
                return None
            # solve W
            for r in range(rank):
                # W[:,r] free; minimize sum_{i,j,k} (T_ijk - U_ijr V_jkr W_kir)^2
                # For fixed U[:,r], V[:,r], the optimal W_kir:
                # sum_{i,j} (U_ijr V_jkr) W_kir = sum_{i,j} T_ijk U_ijr V_jkr
                # => W_kir = sum_{i,j} U_ijr V_jkr T_ijk / sum_{i,j} (U_ijr V_jkr)^2
                num = np.zeros(p * n)
                den = np.zeros(p * n)
                for i in range(n):
                    for j in range(m):
                        uij = U[i * m + j, r]
                        if uij == 0:
                            continue
                        for k in range(p):
                            vjk = V[j * p + k, r]
                            idx = k * n + i
                            num[idx] += uij * vjk
                            den[idx] += uij * uij * vjk * vjk
                mask = den > 1e-12
                W[mask, r] = num[mask] / den[mask]
                W[~mask, r] = 0.0
                W[:, r] = round_half(W[:, r])
            # solve V
            for r in range(rank):
                num = np.zeros(m * p)
                den = np.zeros(m * p)
                for i in range(n):
                    for k in range(p):
                        wki = W[k * n + i, r]
                        if wki == 0:
                            continue
                        for j in range(m):
                            uij = U[i * m + j, r]
                            idx = j * p + k
                            num[idx] += wki * uij
                            den[idx] += wki * wki * uij * uij
                mask = den > 1e-12
                V[mask, r] = num[mask] / den[mask]
                V[~mask, r] = 0.0
                V[:, r] = round_half(V[:, r])
            # solve U
            for r in range(rank):
                num = np.zeros(n * m)
                den = np.zeros(n * m)
                for j in range(m):
                    for k in range(p):
                        vjk = V[j * p + k, r]
                        if vjk == 0:
                            continue
                        for i in range(n):
                            wki = W[k * n + i, r]
                            idx = i * m + j
                            num[idx] += vjk * wki
                            den[idx] += vjk * vjk * wki * wki
                mask = den > 1e-12
                U[mask, r] = num[mask] / den[mask]
                U[~mask, r] = 0.0
                U[:, r] = round_half(U[:, r])
        return U, V, W

    def remove_zero_cols(U, V, W):
        rank = U.shape[1]
        keep = []
        for r in range(rank):
            if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
                keep.append(r)
        return U[:, keep], V[:, keep], W[:, keep]

    # First verify the base decomposition
    if verify(U_mat, V_mat, W_mat):
        best_rank = base_rank
    else:
        # Fallback to a guaranteed-valid schoolbook rank-80
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Refinement loop with multiple restarts
    rng = np.random.default_rng(7)
    while time.time() < deadline - 5.0:
        # random restart: perturb and drop some columns
        U0, V0, W0 = U_mat.copy(), V_mat.copy(), W_mat.copy()
        # drop a random fraction of columns
        keep = np.random.RandomState(0).choice(base_rank, size=int(base_rank * 0.9), replace=False)
        Uc, Vc, Wc = U0[:, keep], V0[:, keep], W0[:, keep]
        result = als_round(Uc, Vc, Wc, iters=4)
        if result is None:
            break
        U1, V1, W1 = result
        U1, V1, W1 = remove_zero_cols(U1, V1, W1)
        if U1.shape[1] < best_rank and verify(U1, V1, W1):
            best_rank = U1.shape[1]
            best_U, best_V, best_W = U1, V1, W1
        # try also from the best-so-far
        if time.time() < deadline - 8.0:
            result2 = als_round(best_U, best_V, best_W, iters=3)
            if result2 is not None:
                U2, V2, W2 = result2
                U2, V2, W2 = remove_zero_cols(U2, V2, W2)
                if U2.shape[1] < best_rank and verify(U2, V2, W2):
                    best_rank = U2.shape[1]
                    best_U, best_V, best_W = U2, V2, W2

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
