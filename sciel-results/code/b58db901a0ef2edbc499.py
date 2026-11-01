"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    def verify(U, V, W, rank):
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def make_rank27():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a rank-23 decomposition
    random.seed(12345)
    best_U, best_V, best_W = None, None, None
    search_deadline = time.time() + 130.0

    # Strategy: Use a structured search based on known algebraic identities
    # Try various densities and patterns
    
    for density in [0.2, 0.3, 0.4, 0.5]:
        if time.time() > search_deadline:
            break
        for attempt in range(50000):
            if time.time() > search_deadline:
                break
            
            U = np.zeros((9, 23), dtype=int)
            V = np.zeros((9, 23), dtype=int)
            W = np.zeros((9, 23), dtype=int)
            
            for r in range(23):
                for idx in range(9):
                    if random.random() < density:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(9):
                    if random.random() < density:
                        V[idx, r] = random.choice([-1, 1])
                for idx in range(9):
                    if random.random() < density:
                        W[idx, r] = random.choice([-1, 1])
            
            if verify(U, V, W, 23):
                best_U, best_V, best_W = U, V, W
                break
        if best_U is not None:
            break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback to schoolbook rank-27
    U, V, W = make_rank27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
