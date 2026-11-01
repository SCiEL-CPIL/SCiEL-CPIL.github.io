"""Matrix multiplication 2x5x6 with rank-54 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank 54 decomposition:
    # 6 Strassen 2x2x2 blocks (rank 7 each = 42)
    #   - j in {0,1} with k in {0,1}, {2,3}, {4,5}  (3 blocks)
    #   - j in {2,3} with k in {0,1}, {2,3}, {4,5}  (3 blocks)
    # 12 schoolbook terms for j=4 (rank 12)
    # Total: 54

    rank = 54
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 54

    col = 0

    def add_strassen_2x2x2(i_base, j_base, k_base, c):
        """Add a 2x2x2 Strassen block. Returns c+7."""
        # A indices: (i, j) -> row i*m+j in U
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)

        # B indices: (j, k) -> row j*p+k in V
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)

        # C indices: (k, i) -> row k*n+i in W
        w00 = k_base * n + i_base
        w01 = (k_base + 1) * n + i_base
        w10 = k_base * n + (i_base + 1)
        w11 = (k_base + 1) * n + (i_base + 1)

        # M1 = (A00 + A11)(B00 + B11)
        U[u00][c] += 1; U[u11][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w11][c] += 1
        c += 1

        # M2 = (A10 + A11)B00
        U[u10][c] += 1; U[u11][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1; W[w11][c] += -1
        c += 1

        # M3 = A00(B01 - B11)
        U[u00][c] += 1
        V[v01][c] += 1; V[v11][c] += -1
        W[w01][c] += 1; W[w11][c] += 1
        c += 1

        # M4 = A11(B10 - B00)
        U[u11][c] += 1
        V[v10][c] += 1; V[v00][c] += -1
        W[w00][c] += 1; W[w10][c] += 1
        c += 1

        # M5 = (A00 + A01)B11
        U[u00][c] += 1; U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] += -1; W[w01][c] += 1
        c += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[u10][c] += 1; U[u00][c] += -1
        V[v00][c] += 1; V[v01][c] += 1
        W[w11][c] += 1
        c += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[u01][c] += 1; U[u11][c] += -1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] += 1
        c += 1

        return c

    def add_schoolbook(i, j, k, c):
        """Add a single schoolbook term. Returns c+1."""
        U[i * m + j][c] += 1
        V[j * p + k][c] += 1
        W[k * n + i][c] += 1
        return c + 1

    # 3 Strassen blocks for j in {0,1}
    for k_base in [0, 2, 4]:
        col = add_strassen_2x2x2(0, 0, k_base, col)

    # 3 Strassen blocks for j in {2,3}
    for k_base in [0, 2, 4]:
        col = add_strassen_2x2x2(0, 2, k_base, col)

    # 12 schoolbook terms for j=4
    for i in range(n):
        for k in range(p):
            col = add_schoolbook(i, 4, k, col)

    assert col == rank, f"Expected col={rank}, got {col}"

    # Quick verification
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    if time.time() < deadline - 5.0:
        if verify(U, V, W):
            return {"U": U, "V": W, "W": W} if False else {"U": U, "V": V, "W": W}
        else:
            # Fallback: pure schoolbook rank 60
            rank_fb = n * m * p
            U_f = [[0.0] * rank_fb for _ in range(n * m)]
            V_f = [[0.0] * rank_fb for _ in range(m * p)]
            W_f = [[0.0] * rank_fb for _ in range(p * n)]
            c_f = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i * m + j][c_f] = 1.0
                        V_f[j * p + k][c_f] = 1.0
                        W_f[k * n + i][c_f] = 1.0
                        c_f += 1
            return {"U": U_f, "V": V_f, "W": W_f}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
