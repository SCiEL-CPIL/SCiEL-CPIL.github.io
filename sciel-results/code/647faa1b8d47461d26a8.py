"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8

    # Decompose into 4 blocks of 2x3x2 (column pairs of B).
    # For each 2x3x2 block, use a rank-7 decomposition (Strassen-like).
    #
    # A is 2x3: rows a0=[a00,a01,a02], a1=[a10,a11,a12]
    # B is 3x2: cols b0=[b00,b10,b20], b1=[b01,b11,b21]
    # C = A*B is 2x2
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    #
    # Rank-7 decomposition:
    # M1 = (a00+a10)*(b00+b01)
    # M2 = (a01+a11)*(b10+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b10
    # M6 = (a02-a12)*b20
    # M7 = a00*b01
    #
    # C[0,0] = (M1+M2+M3)/2 + (M4+M5+M6)/2
    # C[0,1] = (M1+M2+M3)/2 - (M4+M5+M6)/2 + M7
    # C[1,0] = (M1+M2+M3)/2 - (M4+M5+M6)/2
    # C[1,1] = (M1+M2+M3)/2 + (M4+M5+M6)/2 - M7
    #
    # Verify:
    # (M1+M2+M3)/2 = (a00+a10)(b00+b01)/2 + (a01+a11)(b10+b11)/2 + (a02+a12)(b20+b21)/2
    # (M4+M5+M6)/2 = (a00-a10)b00/2 + (a01-a11)b10/2 + (a02-a12)b20/2
    #
    # C[0,0] = [a00(b00+b01) + a10(b00+b01) + a01(b10+b11) + a11(b10+b11) + a02(b20+b21) + a12(b20+b21)]/4
    #        + [a00b00 - a10b00 + a01b10 - a11b10 + a02b20 - a12b20]/4
    # = [a00b00 + a00b01 + a10b00 + a10b01 + a01b10 + a01b11 + a11b10 + a11b11 + a02b20 + a02b21 + a12b20 + a12b21
    #    + a00b00 - a10b00 + a01b10 - a11b10 + a02b20 - a12b20]/4
    # = [2a00b00 + a00b01 + a10b01 + 2a01b10 + a01b11 + a11b11 + 2a02b20 + a02b21 + a12b21]/4
    #
    # Hmm, that's not matching. Let me redo this more carefully.
    #
    # Actually, let me just use the standard Strassen approach for 2x2x2 embedded in this.
    # For 2x2x2, Strassen gives rank 7. For 2x3x2, we need to handle the extra column.
    #
    # Let me try a different approach. Use rank 12 schoolbook per block, total 48.
    # But maybe I can do better by sharing across blocks.
    #
    # Actually, let me just implement the schoolbook decomposition which is guaranteed correct.
    rank = n * m * p  # 48

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
# EVOLVE-BLOCK-END
