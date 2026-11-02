"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 140  # seconds, safety margin before 150s kill
    n, m, p = 2, 4, 4
    r = 28

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Four independent 2x2 Strassen products (rank 7 each, total 28)
    # Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block 1: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # Block 3: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]

    def _add_strassen(off, ar0, ac0, br0, bc0, cr0, cc0):
        a00 = ar0 * m + (ac0 + 0)
        a01 = ar0 * m + (ac0 + 1)
        a10 = (ar0 + 1) * m + (ac0 + 0)
        a11 = (ar0 + 1) * m + (ac0 + 1)

        b00 = br0 * p + (bc0 + 0)
        b01 = br0 * p + (bc0 + 1)
        b10 = (br0 + 1) * p + (bc0 + 0)
        b11 = (br0 + 1) * p + (bc0 + 1)

        c00 = cc0 * n + cr0
        c01 = (cc0 + 1) * n + cr0
        c10 = cc0 * n + (cr0 + 1)
        c11 = (cc0 + 1) * n + (cr0 + 1)

        # M1 = (a00+a11)(b00+b11)
        U[a00][off] += 1
        U[a11][off] += 1
        V[b00][off] += 1
        V[b11][off] += 1
        W[c00][off] += 1
        W[c11][off] += 1

        # M2 = (a10+a11)b00
        U[a10][off+1] += 1
        U[a11][off+1] += 1
        V[b00][off+1] += 1
        W[c10][off+1] += 1
        W[c11][off+1] -= 1

        # M3 = a00(b01-b11)
        U[a00][off+2] += 1
        V[b01][off+2] += 1
        V[b11][off+2] -= 1
        W[c01][off+2] += 1
        W[c11][off+2] += 1

        # M4 = a11(b10-b00)
        U[a11][off+3] += 1
        V[b10][off+3] += 1
        V[b00][off+3] -= 1
        W[c00][off+3] += 1
        W[c10][off+3] += 1

        # M5 = (a00+a01)b11
        U[a00][off+4] += 1
        U[a01][off+4] += 1
        V[b11][off+4] += 1
        W[c00][off+4] -= 1
        W[c01][off+4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[a10][off+5] += 1
        U[a00][off+5] -= 1
        V[b00][off+5] += 1
        V[b01][off+5] += 1
        W[c11][off+5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[a01][off+6] += 1
        U[a11][off+6] -= 1
        V[b10][off+6] += 1
        V[b11][off+6] += 1
        W[c00][off+6] += 1

    _add_strassen(0, 0, 0, 0, 0, 0, 0)
    _add_strassen(7, 0, 2, 2, 0, 0, 0)
    _add_strassen(14, 0, 0, 0, 2, 0, 2)
    _add_strassen(21, 0, 2, 2, 2, 0, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
