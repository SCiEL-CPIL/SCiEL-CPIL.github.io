import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60

    def schoolbook_decomp():
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
        return U, V, W, rank

    def verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def to_list(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook_decomp()
    best_rank = rank
    best_result = to_list(U, V, W)

    def try_rank(target_rank, max_iters=80):
        """Try to find a decomposition of given rank using alternating least squares."""
        if target_rank < 1 or time.time() > deadline - 10:
            return None

        np.random.seed(0)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1

        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Solve for W given U, V
            for i in range(n):
                for k in range(p):
                    A = np.zeros(target_rank)
                    for j in range(m):
                        A += U[i*m+j, :] * V[j*p+k, :]
                    if np.linalg.norm(A) > 1e-12:
                        W[k*n+i, :] = np.linalg.lstsq(A.reshape(1, -1), np.array([1.0]), rcond=None)[0]

            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, target_rank))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.matrix_rank(B) > 0:
                        U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]

            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    C = np.zeros((n, target_rank))
                    for i in range(n):
                        C[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.matrix_rank(C) > 0:
                        V[j*p+k, :] = np.linalg.lstsq(C, np.ones(n), rcond=None)[0]

            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0)**2

            if err < 1e-12:
                if verify(U, V, W):
                    return U, V, W, target_rank
                break
            if err > prev_err * 1.01:
                break
            prev_err = err

        if verify(U, V, W):
            return U, V, W, target_rank
        return None

    # Try decreasing ranks from schoolbook-1 downward
    for target_rank in range(schoolbook_rank - 1, 35, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(target_rank)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            best_result = to_list(U, V, W)
            break

    return best_result
# EVOLVE-BLOCK-END
