"""Program for matrix multiplication 2x5x6 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Use a 2x2x2 Strassen-like decomposition for the 2x4x4 sub-tensor
    # (j in {0,1,2,3}, k in {0,1,2,3}) and schoolbook for the remaining parts.
    # 
    # For 2x4x4:
    #   Split i into {0,1}, j into {0,1}x{2,3}, k into {0,1}x{2,3}
    #   We can view this as 2x(2x2)x(2x2)
    #   Using Strassen on the 2x2x2 blocks: 4 blocks of rank 7 = 28
    #   But we can do better with a direct 2x4x4 algorithm
    #
    # Actually, let's use the known result: 2x4x4 has rank 20 (or we can use 4x4x4 Strassen)
    # For 2x4x4, we can split j={0,1,2,3} and k={0,1,2,3}
    # Use 2x2 Strassen blocks: 
    #   j={0,1},k={0,1}: rank 7
    #   j={0,1},k={2,3}: rank 7  
    #   j={2,3},k={0,1}: rank 7
    #   j={2,3},k={2,3}: rank 7
    #   Total for 2x4x4: 28
    #   Plus j={4},k={0..5}: 2*1*6 = 12
    #   Plus j={0..3},k={4,5}: need to handle
    #
    # Let me reconsider: 2x5x6
    # Split j={0,1,2,3,4}, k={0,1,2,3,4,5}
    # 
    # Block 1: j={0,1,2,3}, k={0,1,2,3}: 2x4x4
    # Block 2: j={0,1,2,3}, k={4,5}: 2x4x2
    # Block 3: j={4}, k={0,1,2,3,4,5}: 2x1x6
    #
    # For 2x4x4: Use 4 Strassen 2x2x2 blocks -> rank 28
    # For 2x4x2: 
    #   j={0,1},k={4,5}: Strassen 2x2x2 -> rank 7
    #   j={2,3},k={4,5}: Strassen 2x2x2 -> rank 7
    #   Total: 14
    # For 2x1x6: schoolbook -> rank 12
    # Total: 28 + 14 + 12 = 54
    #
    # Wait, that's the same as before. Let me try a different approach.
    #
    # Alternative: Use the fact that 2x4x4 can be done in rank 20
    # (known result for 2x4x4 matrix multiplication)
    # Then: 20 + 14 + 12 = 46
    #
    # Let me implement a 2x4x4 algorithm with rank 20.
    #
    # Actually, the known rank for 2x4x4 is 20. Let me use that.
    #
    # For now, let's use the 54 rank approach but verify it works.
    # Then we can try to improve.

    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_2x2x2(i_base, j_base, k_base, c):
        """Add a 2x2x2 Strassen block. Returns c+7."""
        u00 = i_base * m + j_base
        u01 = i_base * m + (j_base + 1)
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + (j_base + 1)
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
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

    # Part 1: Strassen blocks for j in {0,1}, k in {0,1,2,3,4,5}
    for k_base in [0, 2, 4]:
        col = add_strassen_2x2x2(0, 0, k_base, col)

    # Part 2: Strassen blocks for j in {2,3}, k in {0,1,2,3,4,5}
    for k_base in [0, 2, 4]:
        col = add_strassen_2x2x2(0, 2, k_base, col)

    # Part 3: Schoolbook for j in {4}, k in {0..5}
    for i in range(n):
        for k in range(p):
            col = add_schoolbook(i, 4, k, col)

    assert col == rank, f"Expected col={rank}, got {col}"

    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
