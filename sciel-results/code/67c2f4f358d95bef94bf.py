"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin under 120s
    
    # Schoolbook decomposition of rank n*m*p
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
    
    # Verify correctness
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.outer(np.outer(U[:, c], V[:, c]), W[:, c]).reshape(n, m, p)
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        raise ValueError("Schoolbook decomposition verification failed")
    
    # Try quick rank reduction: check if any single term is redundant
    # This is unlikely to succeed but costs little
    for c in range(rank):
        if time.time() > deadline:
            break
        T_test = np.zeros((n, m, p))
        for cc in range(rank):
            if cc == c:
                continue
            T_test += np.outer(np.outer(U[:, cc], V[:, cc]), W[:, cc]).reshape(n, m, p)
        if np.allclose(T_test, target, atol=1e-10):
            # Remove column c
            mask = np.ones(rank, dtype=bool)
            mask[c] = False
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            rank -= 1
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
