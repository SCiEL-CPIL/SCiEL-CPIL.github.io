"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: rows 0,1 of A (2x2) times rows 0,1 of B (2x10).
    Partition the output columns (10) into five 2-wide blocks.
    For each 2x2 x 2x2 sub-multiplication, use the classic rank-7 Strassen algorithm.
    Subtotal: 5 * 7 = 35.
  - Part 2: row 2 of A (2x1) times row 2 of B (1x10).
    Schoolbook: 2 * 1 * 10 = 20.
  Total rank: 55.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    rank = 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[2 * k0, idx] = 1.0
        W[2 * k1 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[3, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[2 * k0 + 1, idx] = 1.0
        W[2 * k1 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0, idx] = 1.0
        V[k1, idx] = 1.0
        V[10 + k1, idx] = -1.0
        W[2 * k1, idx] = 1.0
        W[2 * k1 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[4, idx] = 1.0
        V[10 + k0, idx] = 1.0
        V[k0, idx] = -1.0
        W[2 * k0, idx] = 1.0
        W[2 * k0 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0, idx] = 1.0
        U[1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[2 * k0, idx] = -1.0
        W[2 * k1, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[3, idx] = 1.0
        U[0, idx] = -1.0
        V[k0, idx] = 1.0
        V[k1, idx] = 1.0
        W[2 * k1 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1, idx] = 1.0
        U[4, idx] = -1.0
        V[10 + k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[2 * k0, idx] = 1.0
        idx += 1

    for i in range(2):
        for k in range(10):
            U[6 + i, idx] = 1.0
            V[20 + k, idx] = 1.0
            W[2 * k + i, idx] = 1.0
            idx += 1

    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
