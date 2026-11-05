"""
Tensor decomposition for 2x3 by 3x10 matrix multiplication.
Rank 55 using Strassen on 2x2 sub-blocks + schoolbook for remaining.
"""
import time
import numpy as np


def construct():
    start_time = time.time()
    n, m, p = 2, 3, 10

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: 5 Strassen blocks for 2x2 x 2x10 (rows 0,1 of A, rows 0,1 of B)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[3, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0, idx] = 1.0
        V[10 + k1, idx] = -1.0
        V[k1, idx] = 1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[4, idx] = 1.0
        V[10 + k0, idx] = -1.0
        V[k0, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0, idx] = 1.0
        U[1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[3, idx] = 1.0
        U[0, idx] = -1.0
        V[k0, idx] = 1.0
        V[k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1, idx] = 1.0
        U[4, idx] = -1.0
        V[10 + k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10 (row 2 of A, row 2 of B)
    for i in range(2):
        for k in range(10):
            U[2 + i, idx] = 1.0
            V[20 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    if time.time() - start_time > 140.0:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")

    n, m, p = 2, 3, 10
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        prod_A = 0.0
        for i in range(n):
            for j in range(m):
                prod_A += U[i * m + j, r] * A[i, j]

        prod_B = 0.0
        for j in range(m):
            for k in range(p):
                prod_B += V[j * p + k, r] * B[j, k]

        val = prod_A * prod_B
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += val * W[k * n + i, r]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
