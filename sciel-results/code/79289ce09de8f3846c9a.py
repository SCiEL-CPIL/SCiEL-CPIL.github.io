import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback (rank 40)
    rank_school = n * m * p  # 40
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5
    # Rank 36: four 2x2x2 Strassen blocks (4*7=28) + one 2x4x1 schoolbook block (8)
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        c += 1

        # M2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        c += 1

        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        c += 1

        # M4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        c += 1

        # M5 = (A00 + A01) * B11
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        c += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        c += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        c += 1

        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj), c] = 1
                V[(j0 + dj) * p + k0, c] = 1
                W[k0 * n + (i0 + di), c] = 1
                c += 1
        return c

    col = 0
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W, rank):
        """
        Verify that the decomposition exactly decomposes the matrix multiplication tensor.
        For diagonal terms: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        For off-diagonal terms: sum_r U[a, r] * V[b, r] * W[c, r] == 0
        """
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W, total_rank):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
