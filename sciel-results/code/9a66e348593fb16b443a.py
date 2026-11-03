import time
import numpy as np
import random
from scipy.optimize import minimize

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

    # ---- Refinement machinery: random linear-combination rank reduction ----
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

    def polish_decomposition(U, V, W, tol=1e-10, max_iter=100):
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        for iteration in range(max_iter):
            res = residual_vec(U, V, W)
            res_norm = np.linalg.norm(res)
            if res_norm < tol:
                break

            J = build_jacobian(U, V, W)
            try:
                delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
            except Exception:
                return None

            alpha = 1.0
            improved = False
            for _ in range(30):
                x_trial = tensors_to_vec(U, V, W) + alpha * delta
                U_t, V_t, W_t = vec_to_tensors(x_trial, r)
                res_trial = residual_vec(U_t, V_t, W_t)
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

    def build_4x4x3_optimized():
        # 4x4x3 = two 2x4x3 blocks stacked, each 2x4x3 = two 2x2x3 blocks
        # 2x2x3 = 2x2x2 (Strassen rank 7) + 2x2x1 (rank 4 schoolbook) = rank 11
        # Total: 4 blocks * 11 = 44... but we can do better with shared structure
        # Actually: 4x4x3 can be seen as:
        # rows 0-1, cols 0-1, k 0-2: 2x2x3 (rank 11)
        # rows 0-1, cols 2-3, k 0-2: 2x2x3 (rank 11)
        # rows 2-3, cols 0-1, k 0-2: 2x2x3 (rank 11)
        # rows 2-3, cols 2-3, k 0-2: 2x2x3 (rank 11)
        # Total rank 44 for the 4x4x3 part
        # Combined with 4x4x4 (rank 49) = 93 total
        
        U = [[0] * 44 for _ in range(16)]
        V = [[0] * 44 for _ in range(12)]
        W = [[0] * 44 for _ in range(12)]
        
        # Build one 2x2x3 block: rank 11
        # 2x2x2 part: Strassen rank 7
        # 2x2x1 part: rank 4 (schoolbook for 3rd column)
        def build_2x2x3_block():
            U22 = [[0] * 11 for _ in range(4)]
            V22 = [[0] * 11 for _ in range(6)]
            W22 = [[0] * 11 for _ in range(6)]
            
            # 2x2x2 Strassen (cols 0-1, k 0-1)
            U22[0][0] = 1; U22[3][0] = 1
            V22[0][0] = 1; V22[3][0] = 1
            W22[0][0] = 1; W22[3][0] = 1
            
            U22[2][1] = 1; U22[3][1] = 1
            V22[0][1] = 1
            W22[1][1] = 1; W22[3][1] = -1
            
            U22[0][2] = 1
            V22[1][2] = 1; V22[3][2] = -1
            W22[2][2] = 1; W22[3][2] = 1
            
            U22[3][3] = 1
            V22[2][3] = 1; V22[0][3] = -1
            W22[0][3] = 1; W22[1][3] = 1
            
            U22[0][4] = 1; U22[1][4] = 1
            V22[3][4] = 1
            W22[0][4] = -1; W22[2][4] = 1
            
            U22[2][5] = 1; U22[0][5] = -1
            V22[0][5] = 1; V22[1][5] = 1
            W22[3][5] = 1
            
            U22[1][6] = 1; U22[3][6] = -1
            V22[2][6] = 1; V22[3][6] = 1
            W22[0][6] = 1
            
            # 2x2x1 schoolbook (col 2, k 2): rank 4
            for c in range(4):
                U22[c][7 + c] = 1
                V22[c][9 + c] = 1  # j*p+k where j=0..1, k=2 => j*3+2 = 2,3,5,6... wait
                # Actually for 2x2x3: V has shape (2*3, 11) = (6, 11)
                # V[j*3+k]: for k=2, j=0 => V[2], j=1 => V[5]
                # W has shape (3*2, 11) = (6, 11)
                # W[k*2+i]: for k=2, i=0 => W[4], i=1 => W[5]
                pass
            
            # Let me redo the 2x2x1 part properly
            # For 2x2x1: A(2x2) * B(2x1) = C(2x1)
            # U: (2*2, r) = (4, r)
            # V: (2*1, r) = (2, r)  
            # W: (1*2, r) = (2, r)
            # But we're embedding into 2x2x3, so:
            # U[i*2+j] for i,j in 0..1
            # V[j*3+k] for j in 0..1, k in 0..2
            # W[k*2+i] for k in 0..2, i in 0..1
            
            # 2x2x1 schoolbook: 4 terms
            # U[i*2+j], V[j*3+2], W[2*2+i]
            U22[0][7] = 1  # i=0,j=0
            V22[2][7] = 1  # j=0,k=2
            W22[4][7] = 1  # k=2,i=0
            
            U22[1][8] = 1  # i=0,j=1
            V22[5][8] = 1  # j=1,k=2
            W22[4][8] = 1  # k=2,i=0
            
            U22[2][9] = 1  # i=1,j=0
            V22[2][9] = 1  # j=0,k=2
            W22[5][9] = 1  # k=2,i=1
            
            U22[3][10] = 1  # i=1,j=1
            V22[5][10] = 1  # j=1,k=2
            W22[5][10] = 1  # k=2,i=1
            
            return U22, V22, W22
        
        U22, V22, W22 = build_2x2x3_block()
        
        # Now place 4 blocks
        # Block (bi, bj) for bi in 0..1, bj in 0..1
        # U: row i*4+j where i = bi*2+i2, j = bj*2+j2
        # V: row j*7+k where j = bj*2+j2, k = k2
        # W: row k*4+i where k = k2, i = bi*2+i2
        for bi in range(2):
            for bj in range(2):
                block_offset = (bi * 2 + bj) * 11
                for i2 in range(2):
                    for j2 in range(2):
                        i = bi * 2 + i2
                        j = bj * 2 + j2
                        for c in range(11):
                            if U22[i2 * 2 + j2][c] != 0:
                                U[i * 4 + j][block_offset + c] = U22[i2 * 2 + j2][c]
                for j2 in range(2):
                    for k2 in range(3):
                        j = bj * 2 + j2
                        for c in range(11):
                            if V22[j2 * 3 + k2][c] != 0:
                                V[j * 7 + k2][block_offset + c] = V22[j2 * 3 + k2][c]
                for k2 in range(3):
                    for i2 in range(2):
                        i = bi * 2 + i2
                        for c in range(11):
                            if W22[k2 * 2 + i2][c] != 0:
                                W[k2 * 4 + i][block_offset + c] = W22[k2 * 2 + i2][c]
        
        return U, V, W

    def build_4x4x7_optimized():
        U4, V4, W4 = kronecker_decompose_4x4x4()
        U3, V3, W3 = build_4x4x3_optimized()
        r4 = len(U4[0])
        r3 = len(U3[0])
        r_total = r4 + r3
        
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
        for i in range(16):
            for r in range(r3):
                U[i][col_offset + r] = U3[i][r]
        for j in range(4):
            for k in range(3):
                for r in range(r3):
                    V[j * 7 + k][col_offset + r] = V3[j * 3 + k][r]
        for k in range(3):
            for i in range(4):
                for r in range(r3):
                    W[k * 4 + i][col_offset + r] = W3[k * 4 + i][r]
        
        return U, V, W

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

    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_opt, V_opt, W_opt = build_4x4x7_optimized()
            r_opt = len(U_opt[0])
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
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

        if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            try:
                polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=50)
                if polished is not None:
                    cur_U, cur_V, cur_W = polished
                    best_result = {
                        "U": cur_U.tolist(),
                        "V": cur_V.tolist(),
                        "W": cur_W.tolist(),
                    }
            except Exception:
                pass

    return best_result
