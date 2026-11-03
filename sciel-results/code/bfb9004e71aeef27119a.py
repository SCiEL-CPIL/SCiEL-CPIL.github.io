"""Strassen-based rank-68 decomposition for 3x3 * 3x8 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]  # 9 x 68
    V = [[0] * rank for _ in range(m * p)]  # 24 x 68
    W = [[0] * rank for _ in range(p * n)]  # 24 x 68

    # We split the multiplication into 4 column-blocks of B (each 3x2).
    # For each block b (0..3):
    #   A top-left 2x2 (rows 0-1, cols 0-1) * B block top 2x2 (rows 0-1, cols 2b..2b+1) -> Strassen rank 7
    #   A top-right 2x1 (rows 0-1, col 2) * B block bottom 1x2 (row 2, cols 2b..2b+1) -> schoolbook rank 4
    #   A bottom-left 1x2 (row 2, cols 0-1) * B block top 2x2 (rows 0-1, cols 2b..2b+1) -> schoolbook rank 4
    #   A bottom-right 1x1 (row 2, col 2) * B block bottom 1x2 (row 2, cols 2b..2b+1) -> schoolbook rank 2
    # Total per block: 7 + 4 + 4 + 2 = 17
    # Total: 4 * 17 = 68

    # Strassen 2x2*2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a10)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a11-a01)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    col = 0
    for b in range(4):
        k0 = 2 * b      # start column of B block
        k1 = 2 * b + 1  # end column of B block

        # --- Strassen for A[0:2,0:2] * B[0:2, k0:k1+1] -> C[0:2, k0:k1+1] ---
        # Local A indices: (0,0),(0,1),(1,0),(1,1)
        # Local B indices: (0,k0),(0,k1),(1,k0),(1,k1)
        # Local C indices: (0,k0),(0,k1),(1,k0),(1,k1)

        # M1: (a00+a11)(b00+b11)
        c = col
        U[0 * 3 + 0][c] = 1  # A(0,0)
        U[1 * 3 + 1][c] = 1  # A(1,1)
        V[0 * 8 + k0][c] = 1  # B(0,k0)
        V[1 * 8 + k1][c] = 1  # B(1,k1)
        W[k0 * 3 + 0][c] = 1  # C(0,k0)
        W[k1 * 3 + 0][c] = 1  # C(0,k1)
        W[k0 * 3 + 1][c] = 1  # C(1,k0)
        W[k1 * 3 + 1][c] = -1  # C(1,k1)
        col += 1

        # M2: (a10+a11)(b00)
        c = col
        U[1 * 3 + 0][c] = 1  # A(1,0)
        U[1 * 3 + 1][c] = 1  # A(1,1)
        V[0 * 8 + k0][c] = 1  # B(0,k0)
        W[k0 * 3 + 1][c] = 1  # C(1,k0)
        W[k1 * 3 + 1][c] = -1  # C(1,k1)
        col += 1

        # M3: (a00)(b10-b11)
        c = col
        U[0 * 3 + 0][c] = 1  # A(0,0)
        V[1 * 8 + k0][c] = 1  # B(1,k0)
        V[1 * 8 + k1][c] = -1  # B(1,k1)
        W[k1 * 3 + 0][c] = 1  # C(0,k1)
        W[k1 * 3 + 1][c] = 1  # C(1,k1)
        col += 1

        # M4: (a11)(b01-b00)
        c = col
        U[1 * 3 + 1][c] = 1  # A(1,1)
        V[0 * 8 + k1][c] = 1  # B(0,k1)
        V[0 * 8 + k0][c] = -1  # B(0,k0)
        W[k0 * 3 + 0][c] = 1  # C(0,k0)
        W[k0 * 3 + 1][c] = 1  # C(1,k0)
        col += 1

        # M5: (a00+a10)(b11)
        c = col
        U[0 * 3 + 0][c] = 1  # A(0,0)
        U[1 * 3 + 0][c] = 1  # A(1,0)
        V[1 * 8 + k1][c] = 1  # B(1,k1)
        W[k0 * 3 + 0][c] = -1  # C(0,k0)
        W[k1 * 3 + 0][c] = 1  # C(0,k1)
        col += 1

        # M6: (a10-a00)(b00+b10)
        c = col
        U[1 * 3 + 0][c] = 1  # A(1,0)
        U[0 * 3 + 0][c] = -1  # A(0,0)
        V[0 * 8 + k0][c] = 1  # B(0,k0)
        V[1 * 8 + k0][c] = 1  # B(1,k0)
        W[k1 * 3 + 1][c] = 1  # C(1,k1)
        col += 1

        # M7: (a11-a01)(b01+b11)
        c = col
        U[1 * 3 + 1][c] = 1  # A(1,1)
        U[0 * 3 + 1][c] = -1  # A(0,1)
        V[0 * 8 + k1][c] = 1  # B(0,k1)
        V[1 * 8 + k1][c] = 1  # B(1,k1)
        W[k0 * 3 + 0][c] = 1  # C(0,k0)
        col += 1

        # --- Schoolbook: A[0:2,2] * B[2, k0:k1+1] -> C[0:2, k0:k1+1] ---
        # A(0,2)*B(2,k0) -> C(0,k0)
        c = col
        U[0 * 3 + 2][c] = 1
        V[2 * 8 + k0][c] = 1
        W[k0 * 3 + 0][c] = 1
        col += 1

        # A(0,2)*B(2,k1) -> C(0,k1)
        c = col
        U[0 * 3 + 2][c] = 1
        V[2 * 8 + k1][c] = 1
        W[k1 * 3 + 0][c] = 1
        col += 1

        # A(1,2)*B(2,k0) -> C(1,k0)
        c = col
        U[1 * 3 + 2][c] = 1
        V[2 * 8 + k0][c] = 1
        W[k0 * 3 + 1][c] = 1
        col += 1

        # A(1,2)*B(2,k1) -> C(1,k1)
        c = col
        U[1 * 3 + 2][c] = 1
        V[2 * 8 + k1][c] = 1
        W[k1 * 3 + 1][c] = 1
        col += 1

        # --- Schoolbook: A[2,0:2] * B[0:2, k0:k1+1] -> C[2, k0:k1+1] ---
        # A(2,0)*B(0,k0) -> C(2,k0)
        c = col
        U[2 * 3 + 0][c] = 1
        V[0 * 8 + k0][c] = 1
        W[k0 * 3 + 2][c] = 1
        col += 1

        # A(2,0)*B(0,k1) -> C(2,k1)
        c = col
        U[2 * 3 + 0][c] = 1
        V[0 * 8 + k1][c] = 1
        W[k1 * 3 + 2][c] = 1
        col += 1

        # A(2,1)*B(1,k0) -> C(2,k0)
        c = col
        U[2 * 3 + 1][c] = 1
        V[1 * 8 + k0][c] = 1
        W[k0 * 3 + 2][c] = 1
        col += 1

        # A(2,1)*B(1,k1) -> C(2,k1)
        c = col
        U[2 * 3 + 1][c] = 1
        V[1 * 8 + k1][c] = 1
        W[k1 * 3 + 2][c] = 1
        col += 1

        # --- Schoolbook: A[2,2] * B[2, k0:k1+1] -> C[2, k0:k1+1] ---
        # A(2,2)*B(2,k0) -> C(2,k0)
        c = col
        U[2 * 3 + 2][c] = 1
        V[2 * 8 + k0][c] = 1
        W[k0 * 3 + 2][c] = 1
        col += 1

        # A(2,2)*B(2,k1) -> C(2,k1)
        c = col
        U[2 * 3 + 2][c] = 1
        V[2 * 8 + k1][c] = 1
        W[k1 * 3 + 2][c] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
