"""Matrix multiplication 4x4 x 4x6 via block-Strassen decomposition, rank 77."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 77

    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77

    # Strassen 2x2x2: A[i,j] indexed as i*2+j
    SA = [
        [1, 0, 0, 1],   # M0: A00+A11
        [0, 0, 1, 1],   # M1: A10+A11
        [1, 0, 0, 0],   # M2: A00
        [0, 0, 0, 1],   # M3: A11
        [1, 1, 0, 0],   # M4: A00+A01
        [-1, 0, 1, 0],  # M5: A10-A00
        [0, 1, 0, -1],  # M6: A01-A11
    ]
    SB = [
        [1, 0, 0, 1],   # M0: B00+B11
        [1, 0, 0, 0],   # M1: B00
        [0, -1, 1, 0],  # M2: B10-B11
        [-1, 1, 0, 0],  # M3: B01-B10
        [0, 0, 0, 1],   # M4: B11
        [1, 0, 1, 0],   # M5: B00+B10
        [0, 1, 0, 1],   # M6: B01+B11
    ]
    # SC[c][s] where c = k*2+i (C[k,i]), s = 0..6
    SC = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]=M0+M3-M4+M6
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]=M2+M4
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]=M1+M3
        [1, -1, 1, 0, 0, 1, 0],  # C[1,1]=M0-M1+M2+M5
    ]

    col = 0
    for s in range(7):
        for t in range(11):
            school = (t < 4)
            is_ = t - 4 if not school else 0

            for i2 in range(2):
                for i1 in range(2):
                    for j2 in range(2):
                        for j1 in range(2):
                            i = 2 * i2 + i1
                            j = 2 * j2 + j1
                            ar = i * 4 + j

                            oa = SA[s][i2 * 2 + j2]
                            if school:
                                pairs = [(0,0),(0,1),(1,0),(1,1)]
                                ia, ib = pairs[t]
                                ii = 1 if (i1, j1) == (ia, ib) else 0
                            else:
                                ii = SA[is_][i1 * 2 + j1]

                            if oa * ii != 0:
                                U[ar][col] += oa * ii

                            for k2 in range(3):
                                for k1 in range(2):
                                    k = 2 * k2 + k1
                                    br = j * 6 + k

                                    ob = SB[s][j2 * 2 + k2]
                                    if school:
                                        if k2 == 0 and k1 == 0:
                                            bp = [(0,0),(1,0)]
                                            bj, bk = bp[t]
                                            ib = 1 if (j1, k1) == (bj, bk) else 0
                                        else:
                                            ib = 0
                                    else:
                                        if k1 == 0:
                                            ib = 0
                                        else:
                                            ib = SB[is_][j1 * 2 + (k1 - 1)]

                                    if ob * ib != 0:
                                        V[br][col] += ob * ib

                                    wr = k * 4 + i
                                    oc = SC[k2 * 2 + i2][s]
                                    if school:
                                        if k2 == 0 and k1 == 0:
                                            cp = [(0,0),(1,0)]
                                            ci, ck = cp[t]
                                            ic = 1 if (i1, k1) == (ci, ck) else 0
                                        else:
                                            ic = 0
                                    else:
                                        if k1 == 0:
                                            ic = 0
                                        else:
                                            ic = SC[(k1 - 1) * 2 + i1][is_]

                                    if oc * ic != 0:
                                        W[wr][col] += oc * ic
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
