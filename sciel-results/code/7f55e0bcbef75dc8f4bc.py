import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_set = set(target_indices)

    def verify(U, V, W, rank, tol=1e-9):
        for (a, b, c) in target_indices:
            s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(s - 1.0) > tol:
                return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        if abs(s) > tol:
                            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for idx, (a, b, c) in enumerate(target_indices):
            U[a, idx] = 1
            V[b, idx] = 1
            W[c, idx] = 1
        return U, V, W

    R = 16
    # Precompute target indicator matrix
    T = np.zeros((6, 9, 6))
    for (a, b, c) in target_indices:
        T[a, b, c] = 1.0

    def alternating_minimization(seed):
        np.random.seed(seed)
        U = np.random.randn(6, R) * 0.5
        V = np.random.randn(9, R) * 0.5
        W = np.random.randn(6, R) * 0.5

        for it in range(500):
            if time.time() > deadline - 20:
                return None

            # Update U: for each a, solve least squares
            for a in range(6):
                # M[r, idx] = V[b,r] * W[c,r] where idx = b*6+c
                # t[idx] = T[a, b, c]
                M = np.zeros((R, 54))
                t = np.zeros(54)
                for b in range(9):
                    for c in range(6):
                        idx = b * 6 + c
                        M[:, idx] = V[b] * W[c]
                        t[idx] = T[a, b, c]
                U[a], _, _, _ = np.linalg.lstsq(M, t, rcond=None)

            # Update V: for each b
            for b in range(9):
                M = np.zeros((R, 36))
                t = np.zeros(36)
                for a in range(6):
                    for c in range(6):
                        idx = a * 6 + c
                        M[:, idx] = U[a] * W[c]
                        t[idx] = T[a, b, c]
                V[b], _, _, _ = np.linalg.lstsq(M, t, rcond=None)

            # Update W: for each c
            for c in range(6):
                M = np.zeros((R, 54))
                t = np.zeros(54)
                for a in range(6):
                    for b in range(9):
                        idx = a * 9 + b
                        M[:, idx] = U[a] * V[b]
                        t[idx] = T[a, b, c]
                W[c], _, _, _ = np.linalg.lstsq(M, t, rcond=None)

            # Check convergence
            max_err = 0.0
            for (a, b, c) in target_indices:
                s = np.dot(U[a], V[b] * W[c])
                max_err = max(max_err, abs(s - 1.0))
            # Sample some non-target entries
            for _ in range(50):
                a = random.randint(0, 5)
                b = random.randint(0, 8)
                c = random.randint(0, 5)
                if (a, b, c) not in target_set:
                    s = np.dot(U[a], V[b] * W[c])
                    max_err = max(max_err, abs(s))

            if max_err < 1e-8:
                return U, V, W

        return None

    for seed in range(20):
        if time.time() > deadline - 25:
            break
        result = alternating_minimization(seed)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
