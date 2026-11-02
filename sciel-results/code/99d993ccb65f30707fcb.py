"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_u(r, i, j, val):
        U[i * m + j][r] += val
    def set_v(r, j, k, val):
        V[j * p + k][r] += val
    def set_w(r, k, i, val):
        W[k * n + i][r] += val

    # Strassen terms for A[0:2, 0:2] * B[0:2, 0:2]
    # M1 = a00 * (b00 + b11)
    set_u(0, 0, 0, 1)
    set_v(0, 0, 0, 1)
    set_v(0, 1, 1, 1)
    set_w(0, 0, 0, 1) # C00
    set_w(0, 1, 0, 1) # C01

    # M2 = (a00 + a10) * (b11 - b01)
    set_u(1, 0, 0, 1)
    set_u(1, 1, 0, 1)
    set_v(1, 1, 1, 1)
    set_v(1, 0, 1, -1)
    set_w(1, 1, 0, 1) # C01
    set_w(1, 1, 1, -1) # C11

    # M3 = (a00 + a01) * b11
    set_u(2, 0, 0, 1)
    set_u(2, 0, 1, 1)
    set_v(2, 1, 1, 1)
    set_w(2, 1, 0, 1) # C01

    # M4 = a11 * (b00 + b11)
    set_u(3, 1, 1, 1)
    set_v(3, 0, 0, 1)
    set_v(3, 1, 1, 1)
    set_w(3, 0, 1, 1) # C10
    set_w(3, 1, 1, 1) # C11

    # M5 = (a10 + a11) * (b01 + b00)
    set_u(4, 1, 0, 1)
    set_u(4, 1, 1, 1)
    set_v(4, 0, 1, 1)
    set_v(4, 0, 0, 1)
    set_w(4, 0, 1, 1) # C10
    set_w(4, 1, 1, 1) # C11

    # M6 = (a01 - a10) * (b00 + b10)
    set_u(5, 0, 1, 1)
    set_u(5, 1, 0, -1)
    set_v(5, 0, 0, 1)
    set_v(5, 1, 0, 1)
    set_w(5, 0, 0, 1) # C00
    set_w(5, 0, 1, -1) # C10

    # M7 = (a00 + a10) * (b10 - b00)
    set_u(6, 0, 0, 1)
    set_u(6, 1, 0, 1)
    set_v(6, 1, 0, 1)
    set_v(6, 0, 0, -1)
    set_w(6, 0, 0, 1) # C00
    set_w(6, 0, 1, -1) # C10

    # Remaining terms for Column 2 of B (index 2) and Row 2 of B (index 2)
    # M8 = a00 * b02
    set_u(7, 0, 0, 1)
    set_v(7, 0, 2, 1)
    set_w(7, 2, 0, 1) # C02

    # M9 = a01 * b12
    set_u(8, 0, 1, 1)
    set_v(8, 1, 2, 1)
    set_w(8, 2, 0, 1) # C02

    # M10 = a10 * b02
    set_u(9, 1, 0, 1)
    set_v(9, 0, 2, 1)
    set_w(9, 2, 1, 1) # C12

    # M11 = a11 * b12
    set_u(10, 1, 1, 1)
    set_v(10, 1, 2, 1)
    set_w(10, 2, 1, 1) # C12

    # M12 = a02 * b22
    set_u(11, 0, 2, 1)
    set_v(11, 2, 2, 1)
    set_w(11, 2, 0, 1) # C02

    # M13 = a12 * b22
    set_u(12, 1, 2, 1)
    set_v(12, 2, 2, 1)
    set_w(12, 2, 1, 1) # C12

    # M14 = a02 * b20
    set_u(13, 0, 2, 1)
    set_v(13, 2, 0, 1)
    set_w(13, 0, 0, 1) # C00

    # M15 = a02 * b21
    set_u(14, 0, 2, 1)
    set_v(14, 2, 1, 1)
    set_w(14, 1, 0, 1) # C01

    # M16 = a12 * b20
    set_u(15, 1, 2, 1)
    set_v(15, 2, 0, 1)
    set_w(15, 0, 1, 1) # C10

    # M17 = a12 * b21
    set_u(16, 1, 2, 1)
    set_v(16, 2, 1, 1)
    set_w(16, 1, 1, 1) # C11

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
