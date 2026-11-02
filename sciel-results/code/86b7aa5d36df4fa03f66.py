import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 30 x rank

    col = 0
    # 7 blocks of 2 columns each (columns 0..13)
    for b in range(7):
        # V indices for this block (B_{row, col} with col = 2*b, 2*b+1)
        v_b00 = 2 * (2 * b)      # B[0, 2b]   -> index 4*b
        v_b10 = 2 * (2 * b) + 1  # B[1, 2b]   -> index 4*b+1
        v_b01 = 2 * (2 * b + 1)  # B[0, 2b+1] -> index 4*b+2
        v_b11 = 2 * (2 * b + 1) + 1  # B[1, 2b+1] -> index 4*b+3

        # W indices for this block (C_{row, col})
        w_c00 = 2 * b            # C[0, 2b]   -> index 2*b
        w_c10 = 2 * b + 1        # C[1, 2b]   -> index 2*b+1
        w_c01 = 2 * (2 * b + 1)  # C[0, 2b+1] -> index 4*b+2
        w_c11 = 2 * (2 * b + 1) + 1  # C[1, 2b+1] -> index 4*b+3

        # Standard Strassen for 2x2 x 2x2 block:
        # U indices: 0:A00, 1:A01, 2:A10, 3:A11

        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        U[2][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        col += 1

        # M3 = A00 * (B01 - B11)
        U[0][col] += 1
        V[v_b01][col] += 1
        V[v_b11][col] -= 1
        W[w_c01][col] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        U[3][col] += 1
        V[v_b10][col] += 1
        V[v_b00][col] -= 1
        W[w_c10][col] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        U[0][col] += 1
        U[1][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col] += 1
        U[0][col] -= 1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] += 1
        U[3][col] -= 1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c01][col] += 1
        col += 1

    # Last column (index 14): schoolbook 4 multiplications
    v_b0 = 2 * 14    # B[0, 14] -> index 28
    v_b1 = 2 * 14 + 1  # B[1, 14] -> index 29
    w_c0 = 14        # C[0, 14] -> index 14
    w_c1 = 15        # C[1, 14] -> index 15

    # A00 * B0
    U[0][col] += 1
    V[v_b0][col] += 1
    W[w_c0][col] += 1
    col += 1

    # A01 * B1
    U[1][col] += 1
    V[v_b1][col] += 1
    W[w_c0][col] += 1
    col += 1

    # A10 * B0
    U[2][col] += 1
    V[v_b0][col] += 1
    W[w_c1][col] += 1
    col += 1

    # A11 * B1
    U[3][col] += 1
    V[v_b1][col] += 1
    W[w_c1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
