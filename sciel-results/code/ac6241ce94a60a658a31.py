"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the middle dimension (3) into 1+2.
  - For the 2x1 by 1x10 part (top): schoolbook rank = 2*1*10 = 20
  - For the 2x2 by 2x10 part (bottom): Strassen (rank 7) for each 2x2x2 block (5 blocks) = 35
  Total rank: 55
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Partition middle dim: 3 = 1 + 2
    # Part 1: row 0 of A (2x1) times row 0 of B (1x10) -> schoolbook rank 20
    # Part 2: rows 1,2 of A (2x2) times rows 1,2 of B (2x10) -> 5 Strassen blocks of 2x2x2

    rank = 20 + 35  # 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Schoolbook for 2x1 x 1x10 (row 0 of A, row 0 of B)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 0, idx] = 1.0
            V[0 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    # Part 2: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    # A sub-block: rows 1,2 of A (indices 1,2 in the 3-dim middle)
    # B sub-block: rows 1,2 of B (indices 1,2 in the 3-dim middle)
    for block in range(5):
        k0 = 2 * block + 1  # first of the 2 middle indices
        k1 = 2 * block + 2  # second of the 2 middle indices

        # Strassen for 2x2 x 2x2:
        # A = [[a11, a12], [a21, a22]] where a11=A[1,k0], a12=A[1,k1], a21=A[2,k0], a22=A[2,k1]
        # B = [[b11, b12], [b21, b22]] where b11=B[k0,2*block], b12=B[k0,2*block+1], etc.
        # Output C = [[c11, c12], [c21, c22]]

        # M1 = (a11 + a22)(b11 + b22)
        U[1 * 3 + k0, idx] = 1.0
        U[2 * 3 + k1, idx] = 1.0
        V[k0 * 10 + 2 * block, idx] = 1.0
        V[k1 * 10 + 2 * block + 1, idx] = 1.0
        W[(2 * block) * 2 + 0, idx] = 1.0
        W[(2 * block + 1) * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (a21 + a22)b11
        U[2 * 3 + k0, idx] = 1.0
        U[2 * 3 + k1, idx] = 1.0
        V[k0 * 10 + 2 * block, idx] = 1.0
        W[(2 * block) * 2 + 1, idx] = 1.0
        W[(2 * block + 1) * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = a11(b12 - b22)
        U[1 * 3 + k0, idx] = 1.0
        V[k0 * 10 + 2 * block + 1, idx] = 1.0
        V[k1 * 10 + 2 * block + 1, idx] = -1.0
        W[(2 * block + 1) * 2 + 0, idx] = 1.0
        W[(2 * block + 1) * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = a22(b21 - b11)
        U[2 * 3 + k1, idx] = 1.0
        V[k1 * 10 + 2 * block, idx] = 1.0
        V[k0 * 10 + 2 * block, idx] = -1.0
        W[(2 * block) * 2 + 0, idx] = 1.0
        W[(2 * block) * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (a11 + a12)b22
        U[1 * 3 + k0, idx] = 1.0
        U[1 * 3 + k1, idx] = 1.0
        V[k1 * 10 + 2 * block + 1, idx] = 1.0
        W[(2 * block) * 2 + 0, idx] = -1.0
        W[(2 * block + 1) * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[2 * 3 + k0, idx] = 1.0
        U[1 * 3 + k0, idx] = -1.0
        V[k0 * 10 + 2 * block, idx] = 1.0
        V[k0 * 10 + 2 * block + 1, idx] = 1.0
        W[(2 * block + 1) * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[1 * 3 + k1, idx] = 1.0
        U[2 * 3 + k1, idx] = -1.0
        V[k1 * 10 + 2 * block, idx] = 1.0
        V[k1 * 10 + 2 * block + 1, idx] = 1.0
        W[(2 * block) * 2 + 0, idx] = 1.0
        idx += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    # Verify correctness
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
