"""Matrix multiplication 4x4x5 rank-minimal decomposition via Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

# Correct Strassen rank-7 decomposition of the 2x2x2 tensor.
# U: [A00, A01, A10, A11], V: [B00, B01, B10, B11], W: [C00, C01, C10, C11]
S2_U = np.array([
    [1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, -1, -1]
], dtype=np.int64)

S2_V = np.array([
    [1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, -1, 1, 0, 1]
], dtype=np.int64)

S2_W = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1]
], dtype=np.int64)


def _build_4x4x4():
    """Build the 49-rank Strassen decomposition for 4x4x4 (n=m=p=4)."""
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    col = 0
    for r in range(7):
        for s in range(7):
            u2 = S2_U[:, r]
            v2 = S2_V[:, r]
            w2 = S2_W[:, r]
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            U4[I * 4 + J, col] = u2[I * 2 + J] * u2[i * 2 + j]
            for J in range(2):
                for K in range(2):
                    for j in range(2):
                        for k in range(2):
                            V4[J * 4 + K, col] = v2[J * 2 + K] * v2[j * 2 + k]
            for K in range(2):
                for I in range(2):
                    for k in range(2):
                        for i in range(2):
                            W4[K * 4 + I, col] = w2[K * 2 + I] * w2[k * 2 + i]
            col += 1
    return U4, V4, W4


def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        U2 = u.reshape(n, m)
        V2 = v.reshape(m, p)
        W2 = w.reshape(p, n)
        for i in range(n):
            for j in range(m):
                uj = U2[i, j]
                if uj == 0:
                    continue
                for k in range(p):
                    T[i, j, k] += uj * V2[j, k] * W2[k, i]
    return np.allclose(T, 1.0, atol=1e-9)


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Part 1: 49-rank Strassen for the 4x4x4 core (first 4 columns of B)
    U4, V4, W4 = _build_4x4x4()

    # Part 2: 16 schoolbook terms for the 5th column of B (k=4)
    U_extra = np.zeros((n * m, 16), dtype=np.int64)
    V_extra = np.zeros((m * p, 16), dtype=np.int64)
    W_extra = np.zeros((p * n, 16), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            U_extra[i * m + j, col] = 1
            V_extra[j * p + 4, col] = 1
            W_extra[4 * n + i, col] = 1
            col += 1

    # Combine
    rank = 49 + 16
    U = np.hstack([U4, U_extra])
    V = np.hstack([V4, V_extra])
    W = np.hstack([W4, W_extra])

    # Verify correctness; fall back to schoolbook if it fails
    if not _verify(U, V, W, n, m, p):
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
