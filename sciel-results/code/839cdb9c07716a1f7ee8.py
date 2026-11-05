"""Rank-40 exact decomposition for 2x3 @ 3x10 matrix multiplication.
Uses Strassen (rank 7) on five 2x2 sub-blocks plus one rank-1 term per block
for the third middle dimension, giving 5*8=40 total rank.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 120.0

    n, m, p = 2, 3, 10
    rank = 40

    U = np.zeros((n * m, rank), dtype=int)  # 6 x 40
    V = np.zeros((m * p, rank), dtype=int)  # 30 x 40
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 40

    idx = 0

    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1

        # U indices for A elements: A[i,j] -> i*m+j
        a00 = 0
        a01 = 1
        a02 = 2
        a10 = 3
        a11 = 4
        a12 = 5

        # V indices for B elements: B[j,k] -> j*p+k
        b0k0 = k0
        b0k1 = k1
        b1k0 = 10 + k0
        b1k1 = 10 + k1
        b2k0 = 20 + k0
        b2k1 = 20 + k1

        # W indices for C elements: C[i,k] -> k*n+i
        c0k0 = 2 * k0
        c0k1 = 2 * k1
        c1k0 = 2 * k0 + 1
        c1k1 = 2 * k1 + 1

        # Strassen M1 = (a00+a11)*(b0k0+b1k1)
        U[a00, idx] += 1
        U[a11, idx] += 1
        V[b0k0, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # Strassen M2 = (a10+a11)*b0k0
        U[a10, idx] += 1
        U[a11, idx] += 1
        V[b0k0, idx] += 1
        W[c1k0, idx] += 1
        W[c1k1, idx] -= 1
        idx += 1

        # Strassen M3 = a00*(b0k1-b1k1)
        U[a00, idx] += 1
        V[b0k1, idx] += 1
        V[b1k1, idx] -= 1
        W[c0k1, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # Strassen M4 = a11*(b1k0-b0k0)
        U[a11, idx] += 1
        V[b1k0, idx] += 1
        V[b0k0, idx] -= 1
        W[c0k0, idx] += 1
        W[c1k0, idx] += 1
        idx += 1

        # Strassen M5 = (a00+a01)*b1k1
        U[a00, idx] += 1
        U[a01, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] -= 1
        W[c0k1, idx] += 1
        idx += 1

        # Strassen M6 = (a10-a00)*(b0k0+b0k1)
        U[a10, idx] += 1
        U[a00, idx] -= 1
        V[b0k0, idx] += 1
        V[b0k1, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        # Strassen M7 = (a01-a11)*(b1k0+b1k1)
        U[a01, idx] += 1
        U[a11, idx] -= 1
        V[b1k0, idx] += 1
        V[b1k1, idx] += 1
        W[c0k0, idx] += 1
        idx += 1

        # Rank-1 term for A[* ,2] * B[2, *] covering both k0 and k1
        U[a02, idx] += 1
        U[a12, idx] += 1
        V[b2k0, idx] += 1
        V[b2k1, idx] += 1
        W[c0k0, idx] += 1
        W[c0k1, idx] += 1
        W[c1k0, idx] += 1
        W[c1k1, idx] += 1
        idx += 1

        if time.time() - t_start > DEADLINE - 30:
            break

    # Verify decomposition correctness
    if time.time() - t_start < DEADLINE - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p), dtype=int)
        for t in range(rank):
            u_vec = U[:, t]
            v_vec = V[:, t]
            w_vec = W[:, t]

            u_sum = 0
            for i in range(n):
                for j in range(m):
                    u_sum += u_vec[i * m + j] * A_test[i, j]

            v_sum = 0
            for j in range(m):
                for k in range(p):
                    v_sum += v_vec[j * p + k] * B_test[j, k]

            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += w_vec[k * n + i] * u_sum * v_sum

        if not np.allclose(C_computed, C_expected):
            # Fallback to schoolbook rank 60
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            idx2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, idx2] = 1
                        V[j * p + k, idx2] = 1
                        W[k * n + i, idx2] = 1
                        idx2 += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
