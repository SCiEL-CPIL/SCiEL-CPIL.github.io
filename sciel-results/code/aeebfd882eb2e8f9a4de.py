import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Strategy: Split 2x5x6 into 2x1x6 (rank 12) + 2x4x6 (rank 42)
    # 2x4x6 = 2x2x6 + 2x2x6, each 2x2x6 = 3 Strassen blocks (rank 21)
    # Total: 12 + 42 = 54, but we can do better.
    #
    # Better: Split j into {0,1} and {2,3,4}, k into {0,1,2} and {3,4,5}
    #   (j01, k012): 2x2x3 -> rank 11 (Strassen 2x2x2 + 3 standard)
    #   (j01, k345): 2x2x3 -> rank 11
    #   (j234, k012): 2x3x3 -> rank 13 (Strassen 2x2x2 + 6 standard)
    #   (j234, k345): 2x3x3 -> rank 13
    # Total: 11 + 11 + 13 + 13 = 48
    #
    # Even better: Use the insight from knowledge: 2x1x6 (rank 12) + 2x4x6
    # 2x4x6: split into 2x2x6 blocks, each using 3 Strassen = rank 21
    # Total: 12 + 21 + 21 = 54
    #
    # Let me try: split j={0,1,2,3,4} into {0,1} and {2,3,4}
    #   j01 x k012: 2x2x3, j01 x k345: 2x2x3
    #   j234 x k012: 2x3x3, j234 x k345: 2x3x3
    # For 2x2x3: Strassen(2x2x2) + 3 standard = 7+3 = 10
    # For 2x3x3: Strassen(2x2x2) + 6 standard = 7+6 = 13
    # Total: 10+10+13+13 = 46
    #
    # Actually, let me use a cleaner approach:
    # j={0,1,2,3,4}, k={0,1,2,3,4,5}
    # Split j into {0,1} and {2,3,4}
    # Split k into {0,1,2} and {3,4,5}
    # 4 blocks: 2x2x3, 2x2x3, 2x3x3, 2x3x3

    def add_strassen_2x2x2(j1, j2, k1, k2, U, V, W, col):
        """Add Strassen 2x2x2 for rows j1,j2 of A and cols k1,k2 of B."""
        u00 = 0 * m + j1
        u01 = 0 * m + j2
        u10 = 1 * m + j1
        u11 = 1 * m + j2
        v00 = j1 * p + k1
        v01 = j1 * p + k2
        v10 = j2 * p + k1
        v11 = j2 * p + k2
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1

        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1

        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1

        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

        return col

    def add_standard(i, j, k, U, V, W, col):
        """Add one standard multiplication term."""
        U[i * m + j][col] += 1
        V[j * p + k][col] += 1
        W[k * n + i][col] += 1
        return col + 1

    # Use the known-good approach: 2x1x6 + 2x4x6
    # 2x1x6: standard multiplications for j=4, all k -> 12 terms
    # 2x4x6: split into two 2x2x6 blocks
    #   2x2x6 for j={0,1}, k={0,1,2,3,4,5}: 3 Strassen blocks = 21
    #   2x2x6 for j={2,3}, k={0,1,2,3,4,5}: 3 Strassen blocks = 21
    # Total: 12 + 21 + 21 = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # 2x1x6: j=4, all i, all k -> standard
    for i in range(n):
        for k in range(p):
            col = add_standard(i, 4, k, U, V, W, col)

    # 2x2x6 for j={0,1}, k={0,1,2,3,4,5}
    # Split k into pairs: (0,1), (2,3), (4,5)
    col = add_strassen_2x2x2(0, 1, 0, 1, U, V, W, col)
    col = add_strassen_2x2x2(0, 1, 2, 3, U, V, W, col)
    col = add_strassen_2x2x2(0, 1, 4, 5, U, V, W, col)

    # 2x2x6 for j={2,3}, k={0,1,2,3,4,5}
    col = add_strassen_2x2x2(2, 3, 0, 1, U, V, W, col)
    col = add_strassen_2x2x2(2, 3, 2, 3, U, V, W, col)
    col = add_strassen_2x2x2(2, 3, 4, 5, U, V, W, col)

    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)

    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify():
        # Should not happen, but fallback to pure standard (rank 60)
        rank_f = n * m * p
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
