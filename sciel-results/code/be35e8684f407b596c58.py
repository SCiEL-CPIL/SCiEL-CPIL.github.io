"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _kron_decomposition():
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = p // block_size
    rank = num_blocks * n * m * block_size
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for kp in range(block_size):
                    U[i * m + j, col] = 1
                    V[j * p + (b * block_size + kp), col] = 1
                    W[(b * block_size + kp) * n + i, col] = 1
                    col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook()
    best_rank = n * m * p
    
    kron_result = _kron_decomposition()
    kron_rank = len(kron_result['U'][0])
    if _verify(np.array(kron_result['U']), np.array(kron_result['V']), np.array(kron_result['W']), n, m, p, kron_rank):
        if kron_rank < best_rank:
            best = kron_result
            best_rank = kron_rank
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]

    random.seed(12345)
    low_bound = max(n * m, m * p, p * n)
    rmax = best_rank

    for r in range(low_bound, rmax + 1):
        if time.time() > deadline:
            break
        trials = 4000
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                for idx in range(n * m):
                    if random.random() < 0.5:
                        U[idx, c] = random.choice(vs)
                for idx in range(m * p):
                    if random.random() < 0.5:
                        V[idx, c] = random.choice(vs)
                for idx in range(p * n):
                    if random.random() < 0.5:
                        W[idx, c] = random.choice(vs)
            if _verify(U, V, W, n, m, p, r):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                return best
    return best
# EVOLVE-BLOCK-END
