"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(actual - expected)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-9, max_err

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist(), rank

    def als_decompose(target_rank, max_iters=200, tol=1e-12, seed=0):
        if time_left() < 5:
            return None, 0.0

        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5

        best_err = float('inf')

        for iteration in range(max_iters):
            if time_left() < 3:
                break

            for i in range(n):
                for j in range(m):
                    row_u = i*m + j
                    M = np.empty((p, target_rank))
                    for k in range(p):
                        M[k] = V[j*p+k] * W[k*n+i]
                    tgt = np.array([1.0 if j == k else 0.0 for k in range(p)])
                    try:
                        U[row_u] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            for j in range(m):
                for k in range(p):
                    row_v = j*p + k
                    M = np.empty((n, target_rank))
                    for i in range(n):
                        M[i] = U[i*m+j] * W[k*n+i]
                    tgt = np.array([1.0 if j == k else 0.0 for i in range(n)])
                    try:
                        V[row_v] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            for k in range(p):
                for i in range(n):
                    row_w = k*n + i
                    M = np.empty((m, target_rank))
                    for j in range(m):
                        M[j] = U[i*m+j] * V[j*p+k]
                    tgt = np.array([1.0 if j == k else 0.0 for j in range(m)])
                    try:
                        W[row_w] = np.linalg.lstsq(M.T, tgt, rcond=None)[0]
                    except:
                        pass

            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = abs(actual - expected)
                        if err > max_err:
                            max_err = err

            if max_err < best_err:
                best_err = max_err
                if max_err < tol:
                    break

        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        ok, final_err = verify(U_list, V_list, W_list, target_rank)
        if ok:
            return (U_list, V_list, W_list), target_rank
        return None, best_err

    def polish(U, V, W, target_rank):
        """Use scipy least_squares to polish the solution."""
        if time_left() < 5:
            return None

        U0 = np.array(U, dtype=np.float64).flatten()
        V0 = np.array(V, dtype=np.float64).flatten()
        W0 = np.array(W, dtype=np.float64).flatten()
        x0 = np.concatenate([U0, V0, W0])

        def residuals(x):
            U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_flat = x[2*n*m*target_rank:].reshape(p*n, target_rank)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        actual = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                        res.append(actual - expected)
            return np.array(res)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            U_polished = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_polished = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W_polished = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
            U_list = U_polished.tolist()
            V_list = V_polished.tolist()
            W_list = W_polished.tolist()
            ok, err = verify(U_list, V_list, W_list, target_rank)
            if ok:
                return (U_list, V_list, W_list)
        except:
            pass
        return None

    def try_rank(target_rank, max_restarts=10):
        for seed in range(max_restarts):
            if time_left() < 5:
                break
            result, err = als_decompose(target_rank, seed=seed)
            if result is not None:
                return result
            # If ALS got close, try polishing
            if err < 1e-4 and time_left() > 10:
                U_tmp, V_tmp, W_tmp = als_decompose(target_rank, seed=seed, max_iters=50)
                if U_tmp is not None:
                    polished = polish(U_tmp, V_tmp, W_tmp, target_rank)
                    if polished is not None:
                        return polished
        return None

    U, V, W, rank = schoolbook()

    for target_rank in range(78, 90):
        if time_left() < 10:
            break
        result = try_rank(target_rank, max_restarts=8)
        if result is not None:
            U, V, W = result
            rank = target_rank
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
