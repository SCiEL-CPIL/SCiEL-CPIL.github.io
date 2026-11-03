"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-44 matrix multiplication decomposition for 2x3 * 3x8.
    Uses Strassen's algorithm for 2x2x2 subblocks and schoolbook for 2x1x1x4 parts.
    Returns a dictionary with matrices U, V, W of shapes (6, 44), (24, 44), (16, 44).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44

    U = np.zeros((n * m, rank))  # (6, 44)
    V = np.zeros((m * p, rank))  # (24, 44)
    W = np.zeros((p * n, rank))  # (16, 44)

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen terms for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        nonlocal col
        # Index mappings
        U_A00 = i1 * m + j1
        U_A01 = i1 * m + j2
        U_A10 = i2 * m + j1
        U_A11 = i2 * m + j2
        V_B00 = j1 * p + k1
        V_B01 = j1 * p + k2
        V_B10 = j2 * p + k1
        V_B11 = j2 * p + k2
        W_C00 = k1 * n + i1
        W_C01 = k2 * n + i1
        W_C10 = k1 * n + i2
        W_C11 = k2 * n + i2

        # M1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U[U_A00, col] = 1
        U[U_A11, col] = 1
        V[V_B00, col] = 1
        V[V_B11, col] = 1
        W[W_C00, col] = 1
        W[W_C11, col] = 1
        col += 1

        # M2 = (A10+A11)B00 -> C10:+1
        U[U_A10, col] = 1
        U[U_A11, col] = 1
        V[V_B00, col] = 1
        W[W_C10, col] = 1
        col += 1

        # M3 = A00(B01-B11) -> C01:+1, C11:+1
        U[U_A00, col] = 1
        V[V_B01, col] = 1
        V[V_B11, col] = -1
        W[W_C01, col] = 1
        W[W_C11, col] = 1
        col += 1

        # M4 = A11(B10-B00) -> C00:+1, C10:+1
        U[U_A11, col] = 1
        V[V_B10, col] = 1
        V[V_B00, col] = -1
        W[W_C00, col] = 1
        W[W_C10, col] = 1
        col += 1

        # M5 = (A00+A01)B11 -> C00:-1, C01:+1
        U[U_A00, col] = 1
        U[U_A01, col] = 1
        V[V_B11, col] = 1
        W[W_C00, col] = -1
        W[W_C01, col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01) -> C11:+1
        U[U_A10, col] = 1
        U[U_A00, col] = -1
        V[V_B00, col] = 1
        V[V_B01, col] = 1
        W[W_C11, col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11) -> C00:+1
        U[U_A01, col] = 1
        U[U_A11, col] = -1
        V[V_B10, col] = 1
        V[V_B11, col] = 1
        W[W_C00, col] = 1
        col += 1

    def add_schoolbook_2x1x4(i1, i2, j, k1, k2):
        """Add schoolbook terms for A[i1:i2+1, j] * B[j, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        nonlocal col
        for i in range(i1, i2 + 1):
            for k in range(k1, k2 + 1):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen)
    add_strassen(0, 1, 0, 1, 0, 1)

    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen)
    add_strassen(0, 1, 0, 1, 2, 3)

    # Block 3: A[0:2,2] * B[2,0:4] -> C[0:2,0:4] (schoolbook 2x1x1x4)
    add_schoolbook_2x1x4(0, 1, 2, 0, 3)

    # Block 4: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6] (Strassen)
    add_strassen(0, 1, 0, 1, 4, 5)

    # Block 5: A[0:2,0:2] * B[0:2,6:8] -> C[0:2,6:8] (Strassen)
    add_strassen(0, 1, 0, 1, 6, 7)

    # Block 6: A[0:2,2] * B[2,4:8] -> C[0:2,4:8] (schoolbook 2x1x1x4)
    add_schoolbook_2x1x4(0, 1, 2, 4, 7)

    # Verify col == 44
    assert col == rank, f"Expected col={rank}, got col={col}"

    # Quick verification
    if time.time() - start_time < deadline - 5:
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0

        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err

        if max_err > 1e-10:
            # Fallback to schoolbook rank 48
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb))
            V_fb = np.zeros((m * p, rank_fb))
            W_fb = np.zeros((p * n, rank_fb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
