"""Program for matrix multiplication 2x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a valid tensor decomposition for 2x3x6 matrix multiplication.
    Uses Strassen-based blocking: split A(2x3) = [A1(2x2) | A2(2x1)],
    B(3x6) = [B1(2x6); B2(1x6)].
    A1*B1 uses 3 Strassen 2x2x2 blocks (rank 7 each) = 21.
    A2*B2 uses schoolbook 2x1x6 = 12.
    Total rank = 33.
    """
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS  # 2, 3, 6

    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Build Strassen-based rank-33 decomposition
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(a_rows, b_row_start, c_col_start):
        """
        Add 7 Strassen factors for A[a_rows, b_row_start:b_row_start+2] * B[b_row_start:b_row_start+2, c_col_start:c_col_start+2]
        -> C[a_rows, c_col_start:c_col_start+2]
        """
        nonlocal col
        ar0, ar1 = a_rows  # row indices in A
        br0 = b_row_start  # row index in B (start of 2-row block)
        cc0 = c_col_start  # col index in C (start of 2-col block)

        # A flat indices (i*m+j):
        a00 = ar0 * m + br0
        a01 = ar0 * m + br0 + 1
        a10 = ar1 * m + br0
        a11 = ar1 * m + br0 + 1

        # B flat indices (j*p+k):
        b00 = br0 * p + cc0
        b01 = br0 * p + cc0 + 1
        b10 = (br0 + 1) * p + cc0
        b11 = (br0 + 1) * p + cc0 + 1

        # W flat indices (k*n+i):
        c00 = cc0 * n + ar0
        c01 = (cc0 + 1) * n + ar0
        c10 = cc0 * n + ar1
        c11 = (cc0 + 1) * n + ar1

        # Strassen:
        # M1 = (a00+a11)(b00+b11) -> C00:+1, C01:+1, C10:+1, C11:+1
        # M2 = (a10+a11)(b00)    -> C10:+1, C11:-1
        # M3 = (a00)(b01-b11)    -> C01:+1, C11:+1
        # M4 = (a11)(b10-b00)    -> C00:-1, C10:+1
        # M5 = (a00+a01)(b11)    -> C00:-1, C01:+1
        # M6 = (a10-a00)(b00+b01)-> C11:+1
        # M7 = (a01-a11)(b10+b11)-> C00:+1, C11:-1

        factors = [
            ({a00: 1.0, a11: 1.0}, {b00: 1.0, b11: 1.0}, {c00: 1.0, c01: 1.0, c10: 1.0, c11: 1.0}),
            ({a10: 1.0, a11: 1.0}, {b00: 1.0}, {c10: 1.0, c11: -1.0}),
            ({a00: 1.0}, {b01: 1.0, b11: -1.0}, {c01: 1.0, c11: 1.0}),
            ({a11: 1.0}, {b10: 1.0, b00: -1.0}, {c00: -1.0, c10: 1.0}),
            ({a00: 1.0, a01: 1.0}, {b11: 1.0}, {c00: -1.0, c01: 1.0}),
            ({a10: 1.0, a00: -1.0}, {b00: 1.0, b01: 1.0}, {c11: 1.0}),
            ({a01: 1.0, a11: -1.0}, {b10: 1.0, b11: 1.0}, {c00: 1.0, c11: -1.0}),
        ]

        for u_dict, v_dict, w_dict in factors:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for idx, val in v_dict.items():
                V[idx][col] += val
            for idx, val in w_dict.items():
                W[idx][col] += val
            col += 1

    # A1*B1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # B[0:2, :] split into 3 blocks of width 2
    add_strassen_block((0, 1), 0, 0)  # B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block((0, 1), 0, 2)  # B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block((0, 1), 0, 4)  # B[0:2, 4:6] -> C[0:2, 4:6]

    # A2*B2: A[:, 2] * B[2, :] -> C[:, :]
    # Schoolbook: 2*6 = 12 factors
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1

    assert col == rank, f"Expected {rank} factors, got {col}"

    # Verify
    if not verify_decomposition(U, V, W, rank, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1

    elapsed = time.time() - start_time
    if elapsed > 100:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
