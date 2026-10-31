DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # u[s][x][y] is coeff of X_xy in P_s
    # v[s][x][y] is coeff of Y_xy in P_s
    u = [
        [[1, 0], [0, 1]],  # P1: (x00+x11)
        [[0, 0], [1, 1]],  # P2: (x10+x11)
        [[1, 0], [0, 0]],  # P3: x00
        [[0, 0], [0, 1]],  # P4: x11
        [[1, 1], [0, 0]],  # P5: (x00+x01)
        [[-1, 0], [1, 0]], # P6: (x10-x00)
        [[0, 1], [0, -1]]  # P7: (x01-x11)
    ]
    v = [
        [[1, 0], [0, 1]],  # P1: (y00+y11)
        [[1, 0], [0, 0]],  # P2: y00
        [[0, 1], [0, -1]], # P3: (y01-y11)
        [[-1, 0], [0, 1]], # P4: (y10-y00)
        [[0, 0], [0, 1]],  # P5: y11
        [[1, 1], [0, 0]],  # P6: (y00+y01)
        [[0, 1], [1, 1]]   # P7: (y10+y11)
    ]
    
    # Block coefficients alpha[l][a][b] for X_l = sum alpha A_ab
    alpha = [
        [[1, 0], [0, 1]],  # M1: A00+A11
        [[0, 0], [1, 1]],  # M2: A10+A11
        [[1, 0], [0, 0]],  # M3: A00
        [[0, 0], [0, 1]],  # M4: A11
        [[1, 1], [0, 0]],  # M5: A00+A01
        [[-1, 0], [1, 0]], # M6: A10-A00
        [[0, 1], [0, -1]]  # M7: A01-A11
    ]
    
    # Block coefficients beta[l][a][b] for Y_l = sum beta B_ab
    beta = [
        [[1, 0], [0, 1]],  # M1: B00+B11
        [[1, 0], [0, 0]],  # M2: B00
        [[0, 1], [0, -1]], # M3: B01-B11
        [[-1, 0], [0, 1]], # M4: B10-B00
        [[0, 0], [0, 1]],  # M5: B11
        [[1, 1], [0, 0]],  # M6: B00+B01
        [[0, 1], [1, 1]]   # M7: B10+B11
    ]
    
    # Result coefficients gamma[l][a][b] for C_ab = sum gamma M_l
    gamma = [
        [[1, 0], [0, 0]],  # M1: C00
        [[0, 0], [1, 0]],  # M2: C10
        [[0, 1], [0, 0]],  # M3: C01
        [[1, 0], [0, 0]],  # M4: C00, C10
        [[-1, 1], [0, 0]], # M5: C00, C01
        [[0, 0], [0, 1]],  # M6: C11
        [[1, 0], [0, 0]]   # M7: C00, C11? No.
    ]
    # Let's re-verify gamma
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    gamma = [
        [[1, 0], [0, 0]],  # M1: C00=1, C11=1? No.
    ]
    # Let's build gamma properly.
    gamma = [[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    # M1
    gamma[0][0][0] = 1
    gamma[0][1][1] = 1
    # M2
    gamma[1][1][0] = 1
    gamma[1][1][1] = -1
    # M3
    gamma[2][0][1] = 1
    gamma[2][1][1] = 1
    # M4
    gamma[3][0][0] = 1
    gamma[3][1][0] = 1
    # M5
    gamma[4][0][0] = -1
    gamma[4][0][1] = 1
    # M6
    gamma[5][1][1] = 1
    # M7
    gamma[6][0][0] = 1
    
    # Strassen result coefficients delta[s][x][y] for M_xy = sum delta P_s
    delta = [
        [[1, 0], [0, 0]],  # P1: M00=1, M11=1
        [[0, 0], [1, 0]],  # P2: M10=1, M11=-1
        [[0, 1], [0, 0]],  # P3: M01=1, M11=1
        [[1, 0], [0, 0]],  # P4: M00=1, M10=1
        [[-1, 1], [0, 0]], # P5: M00=-1, M01=1
        [[0, 0], [0, 1]],  # P6: M11=1
        [[1, 0], [0, 0]]   # P7: M00=1
    ]
    # Re-verify delta
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    delta = [
        [[1, 0], [0, 0]],  # P1
        [[0, 0], [1, 0]],  # P2
        [[0, 1], [0, 0]],  # P3
        [[1, 0], [0, 0]],  # P4
        [[-1, 1], [0, 0]], # P5
        [[0, 0], [0, 1]],  # P6
        [[1, 0], [0, 0]]   # P7
    ]
    # M00: P1(1), P4(1), P5(-1), P7(1)
    # M01: P3(1), P5(1)
    # M10: P2(1), P4(1)
    # M11: P1(1), P2(-1), P3(1), P6(1)
    
    # Correct delta
    delta = [
        [[1, 0], [0, 0]],  # P1: M00=1, M11=1
        [[0, 0], [1, 0]],  # P2: M10=1, M11=-1
        [[0, 1], [0, 0]],  # P3: M01=1, M11=1
        [[1, 0], [0, 0]],  # P4: M00=1, M10=1
        [[-1, 1], [0, 0]], # P5: M00=-1, M01=1
        [[0, 0], [0, 1]],  # P6: M11=1
        [[1, 0], [0, 0]]   # P7: M00=1
    ]
    # M00: 1 (P1) + 1 (P4) - 1 (P5) + 1 (P7) = 2? No.
    # M00 = P1 + P4 - P5 + P7.
    # P1: M00=1. P4: M00=1. P5: M00=-1. P7: M00=1.
    # So delta[0][0][0]=1, delta[3][0][0]=1, delta[4][0][0]=-1, delta[6][0][0]=1.
    # My delta array is indexed by s.
    # delta[s][x][y]
    
    delta = [
        [[1, 0], [0, 0]],  # s=0 (P1): M00=1, M11=1
        [[0, 0], [1, 0]],  # s=1 (P2): M10=1, M11=-1
        [[0, 1], [0, 0]],  # s=2 (P3): M01=1, M11=1
        [[1, 0], [0, 0]],  # s=3 (P4): M00=1, M10=1
        [[-1, 1], [0, 0]], # s=4 (P5): M00=-1, M01=1
        [[0, 0], [0, 1]],  # s=5 (P6): M11=1
        [[1, 0], [0, 0]]   # s=6 (P7): M00=1
    ]
    # Check M00: delta[0][0][0]=1, delta[3][0][0]=1, delta[4][0][0]=-1, delta[6][0][0]=1. Sum=2? No.
    # M00 = P1 + P4 - P5 + P7.
    # P1 coeff 1. P4 coeff 1. P5 coeff -1. P7 coeff 1.
    # So delta[0][0][0]=1, delta[3][0][0]=1, delta[4][0][0]=-1, delta[6][0][0]=1.
    # This matches.
    
    for l in range(7):
        for s in range(7):
            c = 7 * l + s
            for x in range(2):
                for y in range(2):
                    # U contributions
                    for a in range(2):
                        for b in range(2):
                            coeff_u = u[s][x][y] * alpha[l][a][b]
                            if coeff_u != 0:
                                i = 2 * a + x
                                j = 2 * b + y
                                U[i * m + j][c] += coeff_u
                                
                    # V contributions
                    for a in range(2):
                        for b in range(2):
                            coeff_v = v[s][x][y] * beta[l][a][b]
                            if coeff_v != 0:
                                j = 2 * a + x
                                k = 2 * b + y
                                V[j * p + k][c] += coeff_v
                                
                    # W contributions
                    for a in range(2):
                        for b in range(2):
                            coeff_w = gamma[l][a][b] * delta[s][x][y]
                            if coeff_w != 0:
                                i = 2 * a + x
                                k = 2 * b + y
                                W[k * n + i][c] += coeff_w
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
