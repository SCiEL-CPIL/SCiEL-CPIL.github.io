import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1] + A[i2,j2])(B[j1,k1] + B[j2,k2])
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M2 = (A[i2,j1] + A[i2,j2]) B[j1,k1]
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1

        # M3 = A[i1,j1] (B[j1,k2] - B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M4 = A[i2,j2] (B[j2,k1] - B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1

        # M5 = (A[i1,j1] + A[i1,j2]) B[j2,k2]
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1

        # M6 = (A[i2,j1] - A[i1,j1])(B[j1,k1] + B[j1,k2])
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M7 = (A[i1,j2] - A[i2,j2])(B[j2,k1] + B[j2,k2])
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # Block 3: A[:, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [0, 1], [4])

    # Block 4: A[:, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [0, 1])

    # Block 5: A[:, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4] (Schoolbook, rank 4)
    add_schoolbook([0, 1], [2], [2, 3])

    # Block 6: A[:, 2:3] * B[2:3, 4:5] -> C[0:2, 4:5] (Schoolbook, rank 2)
    add_schoolbook([0, 1], [2], [4])

    # Verify exactness
    if time.time() < deadline - 5.0:
        res = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

        if not np.allclose(res, 1.0, atol=1e-9):
            # Fallback to schoolbook
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
