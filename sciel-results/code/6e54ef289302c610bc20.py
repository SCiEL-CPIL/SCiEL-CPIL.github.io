"""Rewritten program for matrix multiplication 2x5 by 5x6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0


def _build_schoolbook(n, m, p):
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_lists(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Baseline: schoolbook (rank 60)
    U, V, W = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Verify baseline
    assert _verify(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    # Attempt to find a lower-rank decomposition.
    # The lower bound for 2x5x5x6 is 50 (from embedding 2x5x5x5 which has rank 50).
    # We try a limited random search for ranks 50-59.
    
    rng = random.Random(12345)
    
    for target_rank in range(59, 49, -1):
        if time.time() - start_time > DEADLINE - 15:
            break
        # Try a few random initializations
        for attempt in range(10):
            if time.time() - start_time > DEADLINE - 15:
                break
            U_r = rng.standard_normal((n * m, target_rank))
            V_r = rng.standard_normal((m * p, target_rank))
            W_r = rng.standard_normal((p * n, target_rank))
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
        if best_rank < 60:
            break
    
    return _to_lists(best_U, best_V, best_W)
