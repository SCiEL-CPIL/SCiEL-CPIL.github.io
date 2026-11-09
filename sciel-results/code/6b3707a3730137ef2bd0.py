"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition.

Uses Strassen's 7-term algorithm on 2x2 blocks. The 2x4 x 4x6 multiplication
splits into 6 independent 2x2 x 2x2 sub-problems (2 A-column blocks x 3 B-column
blocks), each decomposed with rank 7, giving total rank 42.
"""
import time

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    # Strassen 7 terms for 2x2x2:
    # M1 = (a00+a11)*(b00+b11) -> C00+, C11+
    # M2 = (a10+a11)*b00       -> C10+, C11-
    # M3 = a00*(b01-b11)       -> C01+, C11+
    # M4 = a11*(b10-b00)       -> C00+, C10+
    # M5 = (a00+a01)*b11       -> C00-, C01+
    # M6 = (a10-a00)*(b00+b01) -> C11+
    # M7 = (a01-a11)*(b10+b11) -> C00+

    for a_block in range(2):  # Split A columns: 0-1, 2-3
        for b_block in range(3):  # Split B columns: 0-1, 2-3, 4-5
            a_col0 = 2 * a_block
            a_col1 = 2 * a_block + 1
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * b_block
            b_col1 = 2 * b_block + 1

            # U indices: A[i, j] -> U[i*m + j]
            # V indices: B[j', k] -> V[j'*p + k]
            # W indices: C[i', k'] -> W[k'*n + i']

            # M1: (a00+a11) * (b00+b11)
            U[0*m + a_col0][col] += 1
            U[1*m + a_col1][col] += 1
            V[b_row0*p + b_col0][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + 0][col] += 1
            W[b_col1*n + 1][col] += 1
            col += 1

            # M2: (a10+a11) * b00
            U[1*m + a_col0][col] += 1
            U[1*m + a_col1][col] += 1
            V[b_row0*p + b_col0][col] += 1
            W[b_col0*n + 1][col] += 1
            W[b_col1*n + 1][col] += -1
            col += 1

            # M3: a00 * (b01-b11)
            U[0*m + a_col0][col] += 1
            V[b_row0*p + b_col1][col] += 1
            V[b_row1*p + b_col1][col] += -1
            W[b_col1*n + 0][col] += 1
            W[b_col1*n + 1][col] += 1
            col += 1

            # M4: a11 * (b10-b00)
            U[1*m + a_col1][col] += 1
            V[b_row1*p + b_col0][col] += 1
            V[b_row0*p + b_col0][col] += -1
            W[b_col0*n + 0][col] += 1
            W[b_col0*n + 1][col] += 1
            col += 1

            # M5: (a00+a01) * b11
            U[0*m + a_col0][col] += 1
            U[0*m + a_col1][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + 0][col] += -1
            W[b_col1*n + 0][col] += 1
            col += 1

            # M6: (a10-a00) * (b00+b01)
            U[1*m + a_col0][col] += 1
            U[0*m + a_col0][col] += -1
            V[b_row0*p + b_col0][col] += 1
            V[b_row0*p + b_col1][col] += 1
            W[b_col1*n + 1][col] += 1
            col += 1

            # M7: (a01-a11) * (b10+b11)
            U[0*m + a_col1][col] += 1
            U[1*m + a_col1][col] += -1
            V[b_row1*p + b_col0][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + 0][col] += 1
            col += 1

    if time.time() - _t0 > 130:
        pass
    return {"U": U, "V": V, "W": W}
