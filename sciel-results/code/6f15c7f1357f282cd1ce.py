import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        if not np.allclose(T_full, T_target, atol=tol):
            return False
        return True

    def try_rank(R, max_iter=300, lr=0.01, time_budget=10.0):
        """Try to find a rank-R decomposition using gradient descent."""
        start = time.time()
        
        # Random initialization
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5

        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iter):
            if time.time() - start > time_budget:
                break
            if time.time() - start_time > time_limit - 5:
                return None

            T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
            E = T_target - T_full
            loss = np.sum(E * E)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-10:
                break

            # Gradients
            grad_U = -2.0 * np.einsum('uvw,vr,wr->ur', E, V, W)
            grad_V = -2.0 * np.einsum('uvw,ur,wr->vr', E, U, W)
            grad_W = -2.0 * np.einsum('uvw,ur,vr->wr', E, U, V)

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

        # Final check
        if check_decomposition(best_U, best_V, best_W, tol=1e-6):
            return {"U": best_U, "V": best_V, "W": best_W}

        return None

    # Try to find a lower rank decomposition
    # Start from R=41 down to R=38
    for R in range(41, 37, -1):
        if time.time() - start_time > time_limit - 10:
            break

        num_restarts = 5
        for _ in range(num_restarts):
            if time.time() - start_time > time_limit - 10:
                break

            result = try_rank(R, max_iter=200, lr=0.02, time_budget=15.0)
            if result is not None:
                return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

    # Fallback to schoolbook
    return get_schoolbook()
