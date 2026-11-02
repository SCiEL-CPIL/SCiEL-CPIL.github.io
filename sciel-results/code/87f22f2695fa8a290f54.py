"""Matrix multiplication 3x5x6 using alternating minimization for rank reduction."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 110

    def time_left():
        return deadline - time.time()

    # Schoolbook initialization
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

    def compute_error(U, V, W):
        e = 0.0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        v = np.dot(U[i * m + j1], V[j2 * p + k] * W[k * n + i])
                        t = 1.0 if j1 == j2 else 0.0
                        e += (v - t) ** 2
        return e

    def alt_min(U, V, W, max_iters=200):
        for it in range(max_iters):
            if time_left() < 20:
                break
            # Solve for U
            for i in range(n):
                for j1 in range(m):
                    A = np.array([V[j2 * p + k] * W[k * n + i] for j2 in range(m) for k in range(p)])
                    t = np.array([1.0 if j2 == j1 else 0.0 for j2 in range(m) for k in range(p)])
                    U[i * m + j1], _, _, _ = np.linalg.lstsq(A, t, rcond=None)
            # Solve for V
            for j2 in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j1] * W[k * n + i] for i in range(n) for j1 in range(m)])
                    t = np.array([1.0 if j1 == j2 else 0.0 for i in range(n) for j1 in range(m)])
                    V[j2 * p + k], _, _, _ = np.linalg.lstsq(A, t, rcond=None)
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j1] * V[j2 * p + k] for j1 in range(m) for j2 in range(m)])
                    t = np.array([1.0 if j1 == j2 else 0.0 for j1 in range(m) for j2 in range(m)])
                    W[k * n + i], _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        return U, V, W

    # Optimize schoolbook
    U, V, W = alt_min(U, V, W)

    # Try to reduce rank
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for target_rank in range(rank - 1, 20, -1):
        if time_left() < 30:
            break
        if target_rank < best_rank:
            break

        U_t = U.copy()
        V_t = V.copy()
        W_t = W.copy()

        # Zero out columns beyond target_rank
        for c in range(target_rank, rank):
            U_t[:, c] = 0
            V_t[:, c] = 0
            W_t[:, c] = 0

        # Optimize
        U_t, V_t, W_t = alt_min(U_t, V_t, W_t, max_iters=100)

        e = compute_error(U_t, V_t, W_t)
        if e < 1e-6:
            best_rank = target_rank
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
            break

    # Return best solution
    U_out = best_U[:, :best_rank]
    V_out = best_V[:, :best_rank]
    W_out = best_W[:, :best_rank]

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END
