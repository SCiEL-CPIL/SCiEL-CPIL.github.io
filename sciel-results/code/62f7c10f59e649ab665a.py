import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W, tol=1e-9):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        actual = float(np.dot(W_arr[k * n + i], U_arr[i * m + j] * V_arr[jp * p + k]))
                        if abs(actual - expected) > tol:
                            return False
        return True

    def build_trivial():
        rank = n * m * p  # 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    def try_find_decomposition(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        rng = np.random.RandomState(seed)

        U_np = rng.randn(n * m, r) * 0.5
        V_np = rng.randn(m * p, r) * 0.5
        W_np = rng.randn(p * n, r) * 0.5

        for it in range(max_iters):
            if elapsed() > deadline - 10:
                break

            # Fix U and V, solve for W
            W_new = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m * m, r))
                    b_vec = np.zeros(m * m)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            for s in range(r):
                                A_mat[idx, s] = U_np[i * m + j, s] * V_np[jp * p + k, s]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    W_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k * n + i, :] = W_row

            # Fix V and W, solve for U
            U_new = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p * m, r))
                    b_vec = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for jp in range(m):
                            for s in range(r):
                                A_mat[idx, s] = W_new[k * n + i, s] * V_np[jp * p + k, s]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    U_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_new[i * m + j, :] = U_row

            # Fix U and W, solve for V
            V_new = np.zeros((m * p, r))
            for jp in range(m):
                for k in range(p):
                    A_mat = np.zeros((n * m, r))
                    b_vec = np.zeros(n * m)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for s in range(r):
                                A_mat[idx, s] = W_new[k * n + i, s] * U_new[i * m + j, s]
                            b_vec[idx] = 1.0 if jp == j else 0.0
                            idx += 1
                    V_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_new[jp * p + k, :] = V_row

            diff_U = np.max(np.abs(U_new - U_np))
            diff_V = np.max(np.abs(V_new - V_np))
            diff_W = np.max(np.abs(W_new - W_np))
            U_np, V_np, W_np = U_new, V_new, W_new

            if diff_U < 1e-10 and diff_V < 1e-10 and diff_W < 1e-10:
                break

        U_list = [[float(U_np[i, s]) for s in range(r)] for i in range(n * m)]
        V_list = [[float(V_np[i, s]) for s in range(r)] for i in range(m * p)]
        W_list = [[float(W_np[i, s]) for s in range(r)] for i in range(p * n)]

        if validate(U_list, V_list, W_list, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}

        return None

    best_result, best_rank = build_trivial()

    for r in range(71, 25, -1):
        if elapsed() > deadline - 20:
            break
        for seed in [r, r + 100, r + 200]:
            if elapsed() > deadline - 15:
                break
            result = try_find_decomposition(r, max_iters=300, seed=seed)
            if result is not None:
                best_result = result
                best_rank = r
                break
        if best_rank == r:
            break

    return best_result
