import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    N = n * m  # 4
    M = m * p  # 20
    P = p * n  # 20

    # Strassen terms for 2x2x2
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

    num_blocks = p // 2  # 5
    R = 7 * num_blocks  # 35

    U = np.zeros((N, R), dtype=float)
    V = np.zeros((M, R), dtype=float)
    W = np.zeros((P, R), dtype=float)

    col = 0
    for q in range(num_blocks):
        for a_vec, b_vec, c_vec in terms:
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]
            V[2*q, col] = b_vec[0]
            V[2*q+1, col] = b_vec[1]
            V[p + 2*q, col] = b_vec[2]
            V[p + 2*q+1, col] = b_vec[3]
            W[4*q, col] = c_vec[0]
            W[4*q+1, col] = c_vec[1]
            W[4*q+2, col] = c_vec[2]
            W[4*q+3, col] = c_vec[3]
            col += 1

    # Quick verification using vectorized operations
    if time.time() < deadline - 5:
        np.random.seed(12345)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B

        U_contrib = np.zeros((n, R))
        for i in range(n):
            U_contrib[i, :] = sum(A[i, j] * U[i*m+j, :] for j in range(m))

        V_contrib = np.zeros((p, R))
        for k in range(p):
            V_contrib[k, :] = sum(B[j, k] * V[j*p+k, :] for j in range(m))

        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] = np.dot(U_contrib[i, :], V_contrib[k, :] * W[k*n+i, :])

        if not np.allclose(C_true, C_decomp, atol=1e-10):
            R_school = n * m * p
            U = np.zeros((N, R_school), dtype=float)
            V = np.zeros((M, R_school), dtype=float)
            W = np.zeros((P, R_school), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
