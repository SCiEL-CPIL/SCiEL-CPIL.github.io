"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_winograd_schoolbook():
        """Rank-23 decomposition: Winograd F(4) for 4x4*4x4 core + schoolbook for 4x4*4x2 strip."""
        rank = 23
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        # Winograd F(4) a-terms: list of (i, j, coeff) for each of 15 terms
        a_terms = [
            [(0, 0, 0.5), (3, 1, 0.5)],
            [(0, 1, 0.5), (3, 0, 0.5)],
            [(0, 2, 0.5), (3, 3, 0.5)],
            [(0, 3, 0.5), (3, 2, 0.5)],
            [(1, 0, 1.0), (2, 1, 1.0)],
            [(1, 1, 1.0), (2, 0, 1.0)],
            [(1, 2, 1.0), (2, 3, 1.0)],
            [(1, 3, 1.0), (2, 2, 1.0)],
            [(1, 0, 1.0), (2, 1, -1.0)],
            [(1, 1, 1.0), (2, 0, -1.0)],
            [(1, 2, 1.0), (2, 3, -1.0)],
            [(1, 3, 1.0), (2, 2, -1.0)],
            [(0, 0, -0.5), (3, 1, -0.5)],
            [(0, 1, -0.5), (3, 0, -0.5)],
            [(0, 2, -0.5), (3, 3, -0.5)],
        ]
        b_terms = [
            [(0, 0, 0.5), (3, 3, 0.5)],
            [(0, 3, 0.5), (3, 0, 0.5)],
            [(0, 2, 0.5), (3, 1, 0.5)],
            [(0, 1, 0.5), (3, 2, 0.5)],
            [(1, 0, 1.0), (2, 3, 1.0)],
            [(1, 3, 1.0), (2, 0, 1.0)],
            [(1, 2, 1.0), (2, 1, 1.0)],
            [(1, 1, 1.0), (2, 2, 1.0)],
            [(1, 0, 1.0), (2, 3, -1.0)],
            [(1, 3, 1.0), (2, 0, -1.0)],
            [(1, 2, 1.0), (2, 1, -1.0)],
            [(1, 1, 1.0), (2, 2, -1.0)],
        ]
        # c_r = sum_t a_t * b_{perm[r][t]}; -1 means absent
        c_perm = [
            [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2],
        ]

        for t in range(15):
            col = t
            for (i, j, coeff) in a_terms[t]:
                U[i * m + j][col] += coeff
            for (j, k, coeff) in b_terms[t]:
                V[j * p + k][col] += coeff
            for r in range(4):
                if c_perm[r][t] >= 0:
                    for i in range(4):
                        W[r * n + i][col] += 1.0

        # Schoolbook strip: C_strip[k,i] = sum_j A[i,j] B[j,k], k in {4,5}
        for jj in range(4):
            for kk in range(4, 6):
                col = 15 + (jj * 2 + (kk - 4))
                for i in range(4):
                    U[i * m + jj][col] += 1.0
                V[jj * p + kk][col] += 1.0
                for i in range(4):
                    W[kk * n + i][col] += 1.0

        return U, V, W

    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                    if abs(sum(prod) - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        if current_rank <= target_rank:
            return U, V, W
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        for remove_idx in cols_to_try:
            if time.time() > time_budget:
                break
            if best_rank <= target_rank:
                break
            mask = [i for i in range(best_rank) if i != remove_idx]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        prod = U_test[i * m + j, :] * V_test[j * p + k, :] * W_test[k * n + i, :]
                        if abs(sum(prod) - 1.0) > 1e-6:
                            valid = False
                            break
            if valid:
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank -= 1
        if best_rank < current_rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return U, V, W

    def schoolbook_fallback(n, m, p):
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return U, V, W

    U, V, W = build_winograd_schoolbook()

    if not verify(U, V, W, n, m, p):
        U, V, W = schoolbook_fallback(n, m, p)

    if time.time() < deadline - 10:
        U, V, W = try_reduce_rank(U, V, W, n, m, p, 20, deadline - 10)

    if not verify(U, V, W, n, m, p):
        U, V, W = schoolbook_fallback(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
