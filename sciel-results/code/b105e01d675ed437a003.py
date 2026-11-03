import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


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

    def build_jacobian(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        u_len = n * m * r
        v_len = m * p * r
        J = np.zeros((n * m * p, u_len + v_len + p * n * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * p + k
                    for c in range(r):
                        uv = V[j * p + k, c] * W[k * n + i, c]
                        vw = U[i * m + j, c] * W[k * n + i, c]
                        wu = U[i * m + j, c] * V[j * p + k, c]
                        J[row, i * m * r + j * r + c] = uv
                        J[row, u_len + j * p * r + k * r + c] = vw
                        J[row, u_len + v_len + k * n * r + i * r + c] = wu
        return J

    def residual_vec(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
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

        res = residual_vec(U_new, V_new, W_new)
        J = build_jacobian(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)

        if not check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
            return None
        return U2.tolist(), V2.tolist(), W2.tolist(), new_r

    def search_2x2x3_decomposition(rng, max_iter=5000):
        """
        Search for a rank-10 decomposition of the 2x2x3 tensor.
        The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
        needs sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k).
        """
        # 2x2x3: U is 4xR, V is 6xR, W is 6xR
        # Indices: U[i*2+j], V[j*3+k], W[k*2+i]
        target_R = 10
        for attempt in range(max_iter):
            # Random initialization
            U = np.random.randn(4, target_R)
            V = np.random.randn(6, target_R)
            W = np.random.randn(6, target_R)
            
            # Iterative refinement
            for _ in range(200):
                # Compute residuals
                res = np.zeros(12)  # 4*3 = 12 entries (2*2*3)
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            s = 0.0
                            for c in range(target_R):
                                s += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                            res[i*2*3 + j*3 + k] = s - 1.0
                
                if np.max(np.abs(res)) < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
                
                # Build Jacobian
                J = np.zeros((12, 4*target_R + 6*target_R + 6*target_R))
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            row = i*2*3 + j*3 + k
                            for c in range(target_R):
                                J[row, i*2*target_R + j*target_R + c] = V[j*3+k, c] * W[k*2+i, c]
                                J[row, 4*target_R + j*3*target_R + k*target_R + c] = U[i*2+j, c] * W[k*2+i, c]
                                J[row, 4*target_R + 6*target_R + k*2*target_R + i*target_R + c] = U[i*2+j, c] * V[j*3+k, c]
                
                try:
                    delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
                    x = np.concatenate([U.flatten(), V.flatten(), W.flatten()]) + delta
                    U = x[:4*target_R].reshape(4, target_R)
                    V = x[4*target_R:4*target_R+6*target_R].reshape(6, target_R)
                    W = x[4*target_R+6*target_R:].reshape(6, target_R)
                except Exception:
                    break
        
        return None

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

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

    # Try the 2x2x3 decomposition approach
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            sub = search_2x2x3_decomposition(rng, max_iter=1000)
            if sub is not None:
                U_sub, V_sub, W_sub = sub
                # Build full 4x4x7 from four 2x2x3 sub-tensors
                # Each 2x2x3 handles a block: (i1,i2) x (j1,j2) x (k1,k2) where k goes 0..2
                # Actually we need to be more careful. The 4x4x7 can be viewed as 2x2 blocks in i,j
                # and the k dimension is 7 = 3+3+1 or similar.
                # Let's try a different decomposition: 4x4x7 = direct sum of 2x2x3, 2x2x3, 2x2x3, 2x2x3, 2x2x2
                # But that's complex. Let's just use the Kronecker result for now.
                pass
        except Exception:
            pass

    # Refinement: iteratively merge columns on the best valid decomposition
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        max_attempts = 400
        attempts = 0
        target = 30
        while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            merged = False
            for n_merge in (3, 2, 1):
                if cur_rank - n_merge < 1:
                    continue
                res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                attempts += 1
                if res is not None:
                    cur_U, cur_V, cur_W, cur_rank = res
                    merged = True
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": cur_U.tolist(),
                            "V": cur_V.tolist(),
                            "W": cur_W.tolist(),
                        }
                    break
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break
            if not merged:
                break

    return best_result
