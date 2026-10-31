"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of Strassen's 2x2 algorithm applied twice to get 4x4.
    # Rank = 7 * 7 = 49.
    n, m, p = DIMENSIONS
    assert n == 4 and m == 4 and p == 4

    # Strassen 2x2 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # Strassen formulas:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 + M5

    # U maps A entries (index i*2+j) to Strassen combinations
    # V maps B entries (index j*2+k) to Strassen combinations
    # W maps C entries (index k*2+i) from Strassen combinations

    # Strassen U for 2x2: row index i*2+j, col index r (0..6)
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]

    # M1 = (a00+a11)*(b00+b11)
    U_S[0][0] = 1  # a00
    U_S[3][0] = 1  # a11
    V_S[0][0] = 1  # b00
    V_S[3][0] = 1  # b11

    # M2 = (a10+a11)*b00
    U_S[2][1] = 1  # a10
    U_S[3][1] = 1  # a11
    V_S[0][1] = 1  # b00

    # M3 = a00*(b01-b11)
    U_S[0][2] = 1  # a00
    V_S[1][2] = 1  # b01
    V_S[3][2] = -1 # b11

    # M4 = a11*(b10-b00)
    U_S[3][3] = 1  # a11
    V_S[2][3] = 1  # b10
    V_S[0][3] = -1 # b00

    # M5 = (a00+a01)*b11
    U_S[0][4] = 1  # a00
    U_S[1][4] = 1  # a01
    V_S[3][4] = 1  # b11

    # M6 = (a10-a00)*(b00+b01)
    U_S[2][5] = 1  # a10
    U_S[0][5] = -1 # a00
    V_S[0][5] = 1  # b00
    V_S[1][5] = 1  # b01

    # M7 = (a01-a11)*(b10+b11)
    U_S[1][6] = 1  # a01
    U_S[3][6] = -1 # a11
    V_S[2][6] = 1  # b10
    V_S[3][6] = 1  # b11

    # W for C entries
    # C00 = M1 - M4 + M6 + M7
    W_S[0][0] = 1  # C00 gets M1
    W_S[0][3] = -1 # C00 gets -M4
    W_S[0][5] = 1  # C00 gets M6
    W_S[0][6] = 1  # C00 gets M7

    # C01 = M3 + M5
    W_S[1][2] = 1  # C01 gets M3
    W_S[1][4] = 1  # C01 gets M5

    # C10 = M2 + M4
    W_S[2][1] = 1  # C10 gets M2
    W_S[2][3] = 1  # C10 gets M4

    # C11 = M1 - M3 + M2 + M5
    W_S[3][0] = 1  # C11 gets M1
    W_S[3][2] = -1 # C11 gets -M3
    W_S[3][1] = 1  # C11 gets M2
    W_S[3][4] = 1  # C11 gets M5

    # Now build 4x4 via Kronecker product
    rank = 7 * 7
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            # U: index (i*4 + j), decompose i = 2*ib + ii, j = 2*jb + ji
            for ib in range(2):
                for ii in range(2):
                    for jb in range(2):
                        for ji in range(2):
                            i = ib * 2 + ii
                            j = jb * 2 + ji
                            # U[i*4+j, r] = U_S[ib*2+jb, ro] * U_S[ii*2+ji, ri]
                            val = U_S[ib * 2 + jb][ro] * U_S[ii * 2 + ji][ri]
                            if val != 0:
                                U[i * 4 + j][r] = val

            # V: index (j*4 + k), decompose j = 2*jb + ji, k = 2*kb + ki
            for jb in range(2):
                for ji in range(2):
                    for kb in range(2):
                        for ki in range(2):
                            j = jb * 2 + ji
                            k = kb * 2 + ki
                            # V[j*4+k, r] = V_S[jb*2+kb, ro] * V_S[ji*2+ki, ri]
                            val = V_S[jb * 2 + kb][ro] * V_S[ji * 2 + ki][ri]
                            if val != 0:
                                V[j * 4 + k][r] = val

            # W: index (k*4 + i), decompose k = 2*kb + ki, i = 2*ib + ii
            for kb in range(2):
                for ki in range(2):
                    for ib in range(2):
                        for ii in range(2):
                            k = kb * 2 + ki
                            i = ib * 2 + ii
                            # W[k*4+i, r] = W_S[kb*2+ib, ro] * W_S[ki*2+ii, ri]
                            val = W_S[kb * 2 + ib][ro] * W_S[ki * 2 + ii][ri]
                            if val != 0:
                                W[k * 4 + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
