"""Tensor decomposition for 2x4 * 4x5 matrix multiplication (rank 36)."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 36)
    V = [[0] * rank for _ in range(m * p)]   # shape (20, 36)
    W = [[0] * rank for _ in range(p * n)]   # shape (10, 36)

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 Strassen terms for (2x2)A x (2x2)B -> (2x2)C."""
        nonlocal col
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1

        v00 = b_r * p + b_c
        v01 = b_r * p + b_c + 1
        v10 = (b_r + 1) * p + b_c
        v11 = (b_r + 1) * p + b_c + 1

        # W index: W[k*n + i, r] where k=output col, i=output row
        w00 = c_c * n + c_r
        w01 = (c_c + 1) * n + c_r
        w10 = c_c * n + c_r + 1
        w11 = (c_c + 1) * n + c_r + 1

        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1

        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1

        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 4 schoolbook terms for (2x2)A x (2x1)B -> (2x1)C."""
        nonlocal col
        u00 = a_r * m + a_c
        u01 = a_r * m + a_c + 1
        u10 = (a_r + 1) * m + a_c
        u11 = (a_r + 1) * m + a_c + 1

        v0 = b_r * p + b_c
        v1 = (b_r + 1) * p + b_c

        w0 = c_c * n + c_r
        w1 = c_c * n + c_r + 1

        # T1: A00 * B0 -> C0
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        # T2: A01 * B1 -> C0
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        # T3: A10 * B0 -> C1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        # T4: A11 * B1 -> C1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    # Partition output columns into: cols 0-1 (4x4 B block), col 2 (4x1), cols 3-4 (4x2)
    # B is 4x5. Split B's columns as [0,1], [2], [3,4].
    # For B cols [0,1]: 4x2 B, split B rows into [0,1] and [2,3].
    #   A cols [0,1] (2x2) x B rows[0,1] cols[0,1] (2x2) -> C rows[0,1] cols[0,1] (2x2): Strassen
    #   A cols [2,3] (2x2) x B rows[2,3] cols[0,1] (2x2) -> C rows[0,1] cols[0,1] (2x2): Strassen
    add_strassen(0, 0, 0, 0, 0, 0)   # A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0, 0)   # A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]

    # For B col [2]: 4x1 B.
    #   A cols [0,1] x B rows[0,1] col[2] -> C rows[0,1] col[2]: 4 terms
    #   A cols [2,3] x B rows[2,3] col[2] -> C rows[0,1] col[2]: 4 terms
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)   # A[0:2,0:2] x B[0:2,2] -> C[0:2,2]
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)   # A[0:2,2:4] x B[2:4,2] -> C[0:2,2]

    # For B cols [3,4]: 4x2 B.
    #   A cols [0,1] x B rows[0,1] cols[3,4] -> C rows[0,1] cols[3,4]: Strassen
    #   A cols [2,3] x B rows[2,3] cols[3,4] -> C rows[0,1] cols[3,4]: Strassen
    add_strassen(0, 0, 0, 3, 0, 3)   # A[0:2,0:2] x B[0:2,3:5] -> C[0:2,3:5]
    add_strassen(0, 2, 2, 3, 0, 3)   # A[0:2,2:4] x B[2:4,3:5] -> C[0:2,3:5]

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
