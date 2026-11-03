"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen coefficients L^U[t][u][v], L^V[t][u][v]
    # t in 1..7, u,v in 0..1
    LU = [[[0]*2 for _ in range(2)] for _ in range(7)]
    LV = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # t=1
    LU[0][0][0] = 1; LU[0][1][1] = 1
    LV[0][0][0] = 1; LV[0][1][1] = 1
    # t=2
    LU[1][1][0] = 1; LU[1][1][1] = 1
    LV[1][0][0] = 1
    # t=3
    LU[2][0][0] = 1
    LV[2][0][1] = 1; LV[2][1][1] = -1
    # t=4
    LU[3][1][1] = 1
    LV[3][1][0] = 1; LV[3][0][0] = -1
    # t=5
    LU[4][0][0] = 1; LU[4][0][1] = 1
    LV[4][1][1] = 1
    # t=6
    LU[5][1][0] = 1; LU[5][0][0] = -1
    LV[5][0][0] = 1; LV[5][0][1] = 1
    # t=7
    LU[6][0][1] = 1; LU[6][1][1] = -1
    LV[6][1][0] = 1; LV[6][1][1] = 1

    # Outer Strassen coefficients cX[s][I][J], cY[s][I][J]
    # s in 1..7, I,J in 0..1
    CX = [[[0]*2 for _ in range(2)] for _ in range(7)]
    CY = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # s=1
    CX[0][0][0] = 1; CX[0][1][1] = 1
    CY[0][0][0] = 1; CY[0][1][1] = 1
    # s=2
    CX[1][1][0] = 1; CX[1][1][1] = 1
    CY[1][0][0] = 1
    # s=3
    CX[2][0][0] = 1
    CY[2][0][1] = 1; CY[2][1][1] = -1
    # s=4
    CX[3][1][1] = 1
    CY[3][1][0] = 1; CY[3][0][0] = -1
    # s=5
    CX[4][0][0] = 1; CX[4][0][1] = 1
    CY[4][1][1] = 1
    # s=6
    CX[5][1][0] = 1; CX[5][0][0] = -1
    CY[5][0][0] = 1; CY[5][0][1] = 1
    # s=7
    CX[6][0][1] = 1; CX[6][1][1] = -1
    CY[6][1][0] = 1; CY[6][1][1] = 1

    # Output coefficients O[s][I][J], R[t][u][v]
    O = [[[0]*2 for _ in range(2)] for _ in range(7)]
    R = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # O (Outer Reconstruction)
    O[0][0][0] = 1; O[0][1][1] = 1
    O[1][1][0] = 1; O[1][1][1] = -1
    O[2][0][1] = 1; O[2][1][1] = 1
    O[3][0][0] = 1; O[3][1][0] = 1
    O[4][0][0] = -1; O[4][0][1] = 1
    O[5][1][1] = 1
    O[6][0][0] = 1; O[6][1][1] = 1
    
    # R (Inner Reconstruction)
    R[0][0][0] = 1; R[0][1][1] = 1
    R[1][1][0] = 1; R[1][1][1] = -1
    R[2][0][1] = 1; R[2][1][1] = 1
    R[3][0][0] = 1; R[3][1][0] = 1
    R[4][0][0] = -1; R[4][0][1] = 1
    R[5][1][1] = 1
    R[6][0][0] = 1; R[6][1][1] = 1

    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for I in range(2):
                for J in range(2):
                    cx = CX[s][I][J]
                    cy = CY[s][I][J]
                    o = O[s][I][J]
                    if cx == 0 and cy == 0 and o == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            lu = LU[t][u][v]
                            lv = LV[t][u][v]
                            rv = R[t][u][v]
                            
                            if lu != 0:
                                i = 2 * I + u
                                j = 2 * J + v
                                U[i * m + j][r] += lu * cx
                            
                            if lv != 0:
                                j = 2 * I + u
                                k = 2 * J + v
                                V[j * p + k][r] += lv * cy
                            
                            if o != 0 and rv != 0:
                                i = 2 * I + u
                                k = 2 * J + v
                                W[k * n + i][r] += o * rv

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
