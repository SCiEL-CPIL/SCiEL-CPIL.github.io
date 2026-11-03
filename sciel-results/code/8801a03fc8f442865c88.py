import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 148.0
SAFETY_MARGIN = 3.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def build_4x4x7_decomposition():
        """Build 4x4*4x7 as 4x4*4x4 (Strassen rank 49) + 4x4*4x3 (schoolbook rank 48)."""
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j * 7 + k][r] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k * 4 + i][r] = W4[k * 4 + i][r]
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    # ---- Vectorized helpers ----
    def tensors_to_vec(U, V, W):
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def build_jacobian_fast(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        u_len = n * m * r
        v_len = m * p * r
        w_len = p * n * r
        total = u_len + v_len + w_len
        n_residuals = n * m * p
        J = np.zeros((n_residuals, total))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * p + k
                    ui = U[i * m + j, :]
                    vk = V[j * p + k, :]
                    wk = W[k * n + i, :]
                    for c in range(r):
                        J[row, i * m * r + j * r + c] = vk[c] * wk[c]
                        J[row, u_len + j * p * r + k * r + c] = ui[c] * wk[c]
                        J[row, u_len + v_len + k * n * r + i * r + c] = ui[c] * vk[c]
        return J

    def residual_vec_fast(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    res[(i * m + j) * p + k] = s - 1.0
        return res

    def random_merge(U, V, W, n_merge, rng, tol=1e-9):
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        new_r = r - n_merge
        cols = list(range(r))
        rng.shuffle(cols)
        kept = cols[:new_r]
        dropped = cols[new_r:]
        M = np.zeros((new_r, r))
        for t, kc in enumerate(kept):
            M[t, kc] = 1.0
            for dc in dropped:
                M[t, dc] = rng.uniform(-1.0, 1.0)
        U_new = U @ M.T
        V_new = V @ M.T
        W_new = W @ M.T
        res = residual_vec_fast(U_new, V_new, W_new)
        J = build_jacobian_fast(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)
        if not check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
            return None
        return U2.tolist(), V2.tolist(), W2.tolist(), new_r

    def polish_decomposition(U, V, W, tol=1e-10, max_iter=100):
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        for iteration in range(max_iter):
            res = residual_vec_fast(U, V, W)
            res_norm = np.linalg.norm(res)
            if res_norm < tol:
                break
            J = build_jacobian_fast(U, V, W)
            try:
                delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
            except Exception:
                return None
            alpha = 1.0
            improved = False
            for _ in range(30):
                x_trial = tensors_to_vec(U, V, W) + alpha * delta
                U_t, V_t, W_t = vec_to_tensors(x_trial, r)
                res_trial = residual_vec_fast(U_t, V_t, W_t)
                trial_norm = np.linalg.norm(res_trial)
                if trial_norm < res_norm:
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
            x_new = tensors_to_vec(U, V, W) + alpha * delta
            U, V, W = vec_to_tensors(x_new, r)
        if not check_valid(U.tolist(), V.tolist(), W.tolist(), tol=tol):
            return None
        return U, V, W

    # ---- MPO-inspired rank reduction ----
    def mpo_restructure(U, V, W, target_rank, rng, max_iter=50):
        """Try MPO-style restructuring to reduce rank."""
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        if r <= target_rank:
            return None

        # Random projection with structured initialization
        for attempt in range(3):
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()

            # Random Gaussian projection
            seed = rng.randint(0, 2**31)
            np.random.seed(seed)
            P = np.random.randn(r, target_rank)
            U_try = U @ P
            V_try = V @ P
            W_try = W @ P

            # Polish
            polished = polish_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(),
                                           tol=1e-10, max_iter=30)
            if polished is not None:
                return polished[0].tolist(), polished[1].tolist(), polished[2].tolist(), target_rank

        return None

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

    # Try the Kronecker-based construction (rank 93)
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Aggressive rank reduction via random merges
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        max_attempts = 800
        attempts = 0
        target = 25
        consecutive_failures = 0
        max_consecutive_failures = 25

        while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            if consecutive_failures >= max_consecutive_failures:
                break

            merged = False
            for n_merge in (5, 4, 3, 2, 1):
                if cur_rank - n_merge < 1:
                    continue
                res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                attempts += 1
                if res is not None:
                    cur_U, cur_V, cur_W, cur_rank = res
                    merged = True
                    consecutive_failures = 0
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": cur_U.tolist(),
                            "V": cur_V.tolist(),
                            "W": cur_W.tolist(),
                        }
                    break
                consecutive_failures += 1
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break

            if not merged and consecutive_failures >= max_consecutive_failures:
                break

        # Final polish pass
        if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            try:
                polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=100)
                if polished is not None:
                    cur_U, cur_V, cur_W = polished
                    best_result = {
                        "U": cur_U.tolist(),
                        "V": cur_V.tolist(),
                        "W": cur_W.tolist(),
                    }
            except Exception:
                pass

    # MPO-inspired rank reduction
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        for target_r in range(max(20, cur_rank - 10), max(15, cur_rank - 25), -1):
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break
            res = mpo_restructure(cur_U, cur_V, cur_W, target_r, rng, max_iter=20)
            if res is not None:
                U_mpo, V_mpo, W_mpo, r_mpo = res
                if r_mpo < best_rank:
                    best_result = {"U": U_mpo, "V": V_mpo, "W": W_mpo}
                    best_rank = r_mpo
                    cur_U = np.array(U_mpo, dtype=float)
                    cur_V = np.array(V_mpo, dtype=float)
                    cur_W = np.array(W_mpo, dtype=float)

    return best_result
