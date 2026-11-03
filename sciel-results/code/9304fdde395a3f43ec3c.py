"""Rewritten program for matrix multiplication 3,5,6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s wall-clock

    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school

    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    min_rank = max(n * p, m * p, n * m)

    # Try to find a valid decomposition at various ranks
    # Use structured search with random perturbations
    for target_rank in range(min_rank, rank_school):
        if time.time() > deadline - 10:
            break

        # Try multiple random initializations with different strategies
        for attempt in range(30):
            if time.time() > deadline - 5:
                break

            # Different initialization strategies
            strategy = attempt % 3

            if strategy == 0:
                # Small integer/half-integer values
                vals = [-1, -0.5, 0, 0.5, 1]
                U = np.random.choice(vals, size=(n * m, target_rank))
                V = np.random.choice(vals, size=(m * p, target_rank))
                W = np.random.choice(vals, size=(p * n, target_rank))
            elif strategy == 1:
                # Slightly larger range
                vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
                U = np.random.choice(vals, size=(n * m, target_rank))
                V = np.random.choice(vals, size=(m * p, target_rank))
                W = np.random.choice(vals, size=(p * n, target_rank))
            else:
                # Sparse initialization with some structure
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Randomly set some entries
                for _ in range(target_rank):
                    idx_u = np.random.randint(0, n * m)
                    idx_v = np.random.randint(0, m * p)
                    idx_w = np.random.randint(0, p * n)
                    U[idx_u, _] = random.choice([-1, 1])
                    V[idx_v, _] = random.choice([-1, 1])
                    W[idx_w, _] = random.choice([-1, 1])

            if _verify_fast(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                return best_result

    return best_result

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    r = U.shape[1]

    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
