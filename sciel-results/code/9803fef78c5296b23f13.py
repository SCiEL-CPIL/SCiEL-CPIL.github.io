import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS  # n=4, m=4, p=9

    # Rank-minimal decomposition for 4x4 x 4x9
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # Rank(B1 part) = 49 (Strassen 4x4x4)
    # Rank(B2 part) = 49 (Strassen 4x4x4)
    # Rank(B3 part) = 16 (Schoolbook 4x4x1)
    # Total Rank = 114

    total_rank = 114
    U = np.zeros((n * m, total_rank))  # (16, 114)
    V = np.zeros((m * p, total_rank))  # (36, 114)
    W = np.zeros((p * n, total_rank))  # (36, 114)

    # Build the 4x4 x 4x4 Strassen decomposition
    U44, V44, W44 = _build_strassen_4x4x4()

    # Build 4x4 x 4x1 schoolbook
    U41, V41, W41 = _build_schoolbook_4x4x1()

    # Part 1: A*B1 (cols 0-3 of B)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]

    # Part 2: A*B2 (cols 4-7 of B)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]

    # Part 3: A*B3 (col 8 of B)
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]

    # Quick validation
    if time.time() < deadline - 2:
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 9)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 9))
        
        # Efficient validation using matrix operations where possible
        # But the tensor structure is explicit, so we iterate
        for r in range(total_rank):
            # Compute linear forms of A
            a_vec = np.zeros(4)
            for i in range(4):
                for k in range(4):
                    a_vec[i] += U[i * 4 + k, r] * A_test[i, k]
            
            # Compute linear forms of B
            b_vec = np.zeros(9)
            for j in range(9):
                for k in range(4):
                    b_vec[j] += V[k * 9 + j, r] * B_test[k, j]
            
            # Add to C
            for i in range(4):
                for j in range(9):
                    C_computed[i, j] += W[j * 4 + i, r] * a_vec[i] * b_vec[j]

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k, idx] = 1
                        V[k * p + j, idx] = 1
                        W[j * n + i, idx] = 1
                        idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_strassen_4x4x4():
    """Build rank-49 Strassen decomposition for 4x4 x 4x4."""
    U2, V2, W2 = _build_strassen_2x2x2()

    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]

    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]

    return U, V, W

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 x 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook_4x4x1():
    """Build rank-16 schoolbook decomposition for 4x4 x 4x1."""
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))

    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U[i * 4 + k, r] = 1
            V[k, r] = 1
            W[i, r] = 1

    return U, V, W
