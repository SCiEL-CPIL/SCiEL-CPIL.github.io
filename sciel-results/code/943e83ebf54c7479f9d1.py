import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Build target tensor indices
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        for (a, b, c) in target_set:
            if abs(T[a, b, c] - 1) > 1e-9:
                return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target_set:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    # Strassen decomposition for 2x4 x 4x4
    # Split: A (2x4) = [A_left(2x2) | A_right(2x2)]
    # B (4x4) = [[B00(2x2), B01(2x2)], [B10(2x2), B11(2x2)]]
    # C (2x4) = [C_left(2x2) | C_right(2x2)]
    # C_left = A_left * B00 + A_right * B10
    # C_right = A_left * B01 + A_right * B11
    # Use Strassen (rank 7) for each 2x2 product
    # Total rank = 7 + 7 + 7 + 7 = 28

    def strassen_decomposition():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 16 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows

        # Strassen 7 products for 2x2 matrix multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6

        # For our problem, we have 4 separate 2x2 multiplications:
        # Product 1: A_left(2x2) * B00(2x2) -> contributes to C_left
        # Product 2: A_right(2x2) * B10(2x2) -> contributes to C_left
        # Product 3: A_left(2x2) * B01(2x2) -> contributes to C_right
        # Product 4: A_right(2x2) * B11(2x2) -> contributes to C_right

        # Each uses Strassen with 7 rank-1 terms.

        # Define Strassen coefficients for a 2x2 matmul:
        # For matrices X (2x2) and Y (2x2) producing Z (2x2):
        # X indices: x00, x10, x01, x11 (row-major: [x00,x01,x10,x11])
        # Y indices: y00, y01, y10, y11
        # Z indices: z00, z01, z10, z11

        # Strassen products:
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)(y00)
        # M3 = (x00)(y10-y11)
        # M4 = (x11)(y00-y10)
        # M5 = (x00+x01)(y11)
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        # z00 = M1+M4-M5+M7
        # z01 = M3+M5
        # z10 = M2+M4
        # z11 = M1-M2+M3+M6

        # Strassen term definitions: (coeffs for X, coeffs for Y, coeffs for Z)
        # Each term is a list of (index, coeff) for each matrix
        strassen_terms = [
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3
            ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (2, 1)]),
            # M4
            ([(3, 1)], [(0, 1), (2, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]),
        ]

        # For each 2x2 block multiplication, we need to map:
        # X (2x2) comes from A: rows i in {0,1}, cols j in some 2-col block
        # Y (2x2) comes from B: rows j in some 2-row block, cols k in some 2-col block
        # Z (2x2) comes from C: rows i in {0,1}, cols k in some 2-col block

        # A is 2x4: A[i, j] where i=0,1 and j=0,1,2,3
        # B is 4x4: B[j, k] where j=0,1,2,3 and k=0,1,2,3
        # C is 2x4: C[i, k] where i=0,1 and k=0,1,2,3

        # Block 1: A_left * B00 -> C_left
        # A_left = A[0:2, 0:2], B00 = B[0:2, 0:2], C_left = C[0:2, 0:2]
        # Block 2: A_right * B10 -> C_left
        # A_right = A[0:2, 2:4], B10 = B[2:4, 0:2], C_left = C[0:2, 0:2]
        # Block 3: A_left * B01 -> C_right
        # A_left = A[0:2, 0:2], B01 = B[0:2, 2:4], C_right = C[0:2, 2:4]
        # Block 4: A_right * B11 -> C_right
        # A_right = A[0:2, 2:4], B11 = B[2:4, 2:4], C_right = C[0:2, 2:4]

        # For each block, define the mapping from 2x2 indices to global indices
        # A 2x2 block: A[i, j_base + dj] where dj in {0,1}
        # B 2x2 block: B[j_base + dj, k_base + dk]
        # C 2x2 block: C[i, k_base + dk]

        # 2x2 index mapping: idx 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # In row-major: (row, col) -> idx = row*2 + col

        blocks = [
            # (a_col_base, b_row_base, b_col_base, c_col_base)
            (0, 0, 0, 0),   # Block 1: A[*,0:2] * B[0:2,0:2] -> C[*,0:2]
            (2, 2, 0, 0),   # Block 2: A[*,2:4] * B[2:4,0:2] -> C[*,0:2]
            (0, 0, 2, 2),   # Block 3: A[*,0:2] * B[0:2,2:4] -> C[*,2:4]
            (2, 2, 2, 2),   # Block 4: A[*,2:4] * B[2:4,2:4] -> C[*,2:4]
        ]

        col = 0
        for a_col_base, b_row_base, b_col_base, c_col_base in blocks:
            for x_coeffs, y_coeffs, z_coeffs in strassen_terms:
                # x_coeffs: list of (local_idx, coeff) for A block
                for (local_idx, coeff) in x_coeffs:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    global_a = lr * m + (a_col_base + lc)
                    U[global_a][col] += coeff

                # y_coeffs: list of (local_idx, coeff) for B block
                for (local_idx, coeff) in y_coeffs:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    global_b = (b_row_base + lr) * p + (b_col_base + lc)
                    V[global_b][col] += coeff

                # z_coeffs: list of (local_idx, coeff) for C block
                for (local_idx, coeff) in z_coeffs:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    global_c = (c_col_base + lc) * n + lr
                    W[global_c][col] += coeff

                col += 1

        return {"U": U, "V": V, "W": W}

    if time.time() < deadline - 10:
        strassen = strassen_decomposition()
        if verify(strassen["U"], strassen["V"], strassen["W"], 28):
            best = strassen
            best_rank = 28
        else:
            # Debug: print which entries fail
            U_np = np.array(strassen["U"], dtype=float)
            V_np = np.array(strassen["V"], dtype=float)
            W_np = np.array(strassen["W"], dtype=float)
            T = np.zeros((n * m, m * p, p * n))
            for r in range(28):
                T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
            # Check target entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if abs(T[a, b, c] - 1) > 1e-9:
                            pass  # would print in debug

    return best
# EVOLVE-BLOCK-END
