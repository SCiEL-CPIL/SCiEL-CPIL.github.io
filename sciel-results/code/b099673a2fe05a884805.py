"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
import time
import numpy as np

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

    def als_decompose(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-12):
        if time_left() < 5:
            return None, 0.0

        U = np.array(U_init, dtype=np.float64)
        V = np.array(V_init, dtype=np.float64)
        W = np.array(W_init, dtype=np.float64)
        
        if U.shape[1] != target_rank:
            return None, 0.0

        best_err = float('inf')

        for iteration in range(max_iters):
            if time_left() < 3:
                break

            # Update U
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

            # Update V
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

            # Update W
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

            # Calculate error
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

    def try_warm_start(target_rank):
        U_school, V_school, W_school, _ = schoolbook()
        U_s = np.array(U_school)
        V_s = np.array(V_school)
        W_s = np.array(W_school)
        
        if target_rank >= 90:
            return None
        
        cols_to_keep = list(range(90))
        num_to_remove = 90 - target_rank
        rng = np.random.RandomState(42)
        
        for attempt in range(30):
            if time_left() < 10:
                break
            
            cols_to_keep = rng.choice(90, size=target_rank, replace=False)
            cols_to_keep.sort()
            
            U_init = U_s[:, cols_to_keep]
            V_init = V_s[:, cols_to_keep]
            W_init = W_s[:, cols_to_keep]
            
            result, err = als_decompose(U_init, V_init, W_init, target_rank, max_iters=100)
            if result is not None:
                return result
            if err < 1e-6:
                result, err = als_decompose(U_init, V_init, W_init, target_rank, max_iters=300)
                if result is not None:
                    return result

        return None

    def try_random_search(target_rank, max_restarts=20):
        for seed in range(max_restarts):
            if time_left() < 5:
                break
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            result, err = als_decompose(U, V, W, target_rank, max_iters=100)
            if result is not None:
                return result
            if err < 1e-6:
                result, err = als_decompose(U, V, W, target_rank, max_iters=300)
                if result is not None:
                    return result

        return None

    U, V, W, rank = schoolbook()

    for target_rank in range(89, 79, -1):
        if time_left() < 15:
            break
        
        result = try_warm_start(target_rank)
        if result is not None:
            U, V, W = result
            rank = target_rank
            break
        
        result = try_random_search(target_rank, max_restarts=10)
        if result is not None:
            U, V, W = result
            rank = target_rank
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
