import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 140

    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    def check_decomposition(U, V, W):
        R = U.shape[1]
        for u_idx, v_idx, w_idx in indices:
            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > 1e-7:
                return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_rank_optimization(R, num_restarts=20):
        if time.time() > deadline - 20:
            return None

        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        def objective(x):
            U = x[:size_U].reshape((n * m, R))
            V = x[size_U:size_U + size_V].reshape((m * p, R))
            W = x[size_U + size_V:].reshape((p * n, R))

            residual = 0.0
            for u_idx, v_idx, w_idx in indices:
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                residual += (val - 1.0) ** 2
            return residual

        from scipy.optimize import minimize

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            x0 = np.random.randn(total_size) * 0.5

            # Phase 1: L-BFGS-B
            try:
                res1 = minimize(objective, x0, method='L-BFGS-B',
                                options={'maxiter': 100, 'gtol': 1e-6})
            except Exception:
                continue

            # Phase 2: Nelder-Mead polish
            if res1.fun < 1.0: # Only if it got reasonably close
                try:
                    res2 = minimize(objective, res1.x, method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res2.fun < 1e-6:
                        U = res2.x[:size_U].reshape((n * m, R))
                        V = res2.x[size_U:size_U + size_V].reshape((m * p, R))
                        W = res2.x[size_U + size_V:].reshape((p * n, R))

                        # Snap
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        if check_decomposition(U_int, V_int, W_int):
                            return U_int, V_int, W_int

                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if check_decomposition(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                except Exception:
                    pass

        return None

    # Try ranks
    for R in range(59, 49, -1):
        if time.time() > deadline - 30:
            break
        result = try_rank_optimization(R, num_restarts=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
