import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140

    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_set = set(target_indices)

    def schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        for idx, (a, b, c) in enumerate(target_indices):
            U[a, idx] = 1
            V[b, idx] = 1
            W[c, idx] = 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

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

    for target_rank in range(rank_schoolbook - 1, 14, -1):
        if time.time() > deadline - 5:
            break
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            U = np.random.randint(-1, 2, (n * m, target_rank)).astype(float)
            V = np.random.randint(-1, 2, (m * p, target_rank)).astype(float)
            W = np.random.randint(-1, 2, (p * n, target_rank)).astype(float)

            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                a, b, c = random.choice(target_indices)
                s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
                err = 1.0 - s
                if abs(err) < 1e-12:
                    continue
                r = random.randint(0, target_rank - 1)
                d = V[b, r] * W[c, r]
                if abs(d) > 1e-8:
                    U[a, r] += err / d
                elif abs(U[a, r] * W[c, r]) > 1e-8:
                    V[b, r] += err / (U[a, r] * W[c, r])
                elif abs(U[a, r] * V[b, r]) > 1e-8:
                    W[c, r] += err / (U[a, r] * V[b, r])

            if verify(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
