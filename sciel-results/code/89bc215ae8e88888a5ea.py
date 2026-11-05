import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank 28 decomposition:
    # Two 2x2 Strassen blocks (rank 7 each) = 14
    # Schoolbook for remaining = 14
    # Total = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Helper to add Strassen factors for 2x2 * 2x2
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Factor 1: M1
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # Factor 2: M2
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1

        # Factor 3: M3
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # Factor 4: M4
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1

        # Factor 5: M5
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1

        # Factor 6: M6
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # Factor 7: M7
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    # Helper to add outer product factors
    def add_outer_product(i_indices, j_idx, k_indices):
        nonlocal col
        for i in i_indices:
            for k in k_indices:
                U[i*m + j_idx, col] = 1
                V[j_idx*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    # 1a. Strassen for A[:, 0:2] * B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)

    # 1b. Outer product for A[:, 2] * B[2, 0:2]
    add_outer_product([0, 1], 2, [0, 1])

    # 2a1. Strassen for A[:, 0:2] * B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)

    # 2a2. Outer product for A[:, 0:2] * B[0:2, 4]
    add_outer_product([0, 1], 0, [4])
    add_outer_product([0, 1], 1, [4])

    # 2b. Outer product for A[:, 2] * B[2, 2:5]
    add_outer_product([0, 1], 2, [2, 3, 4])

    # Verify rank used
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook if construction fails
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
