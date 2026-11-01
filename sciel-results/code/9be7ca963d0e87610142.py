"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # S_A[r][i][j] is coefficient of A[i][j] in the r-th multiplication's A-part
    S_A = [
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, -1], [1, 0]],
        [[0, 1], [-1, 0]] 
    ]
    S_B = [
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]],
        [[0, 1], [-1, 0]],
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]  
    ]
    
    # Sigma[a][b][r] is coefficient of M_r in C[a][b]
    Sigma = [
        [ [1, 0, 0, 1, -1, 0, 1], 
          [0, 0, 1, 0, 1, 0, 0]   
        ],
        [ [0, 1, 0, 1, 0, 0, 0],   
          [1, -1, 1, 0, 0, 1, 0]  
        ]
    ]

    for q in range(7):
        for r in range(7):
            c = 7 * q + r
            
            for I in range(2):
                for J in range(2):
                    coeff_high = S_A[q][I][J]
                    if coeff_high == 0: continue
                    for x in range(2):
                        for y in range(2):
                            coeff_low = S_A[r][x][y]
                            if coeff_low == 0: continue
                            val = coeff_high * coeff_low
                            if val == 0: continue
                            i = 2 * I + x
                            j = 2 * J + y
                            U[i * m + j][c] += val

            for I in range(2):
                for J in range(2):
                    coeff_high = S_B[q][I][J]
                    if coeff_high == 0: continue
                    for x in range(2):
                        for y in range(2):
                            coeff_low = S_B[r][x][y]
                            if coeff_low == 0: continue
                            val = coeff_high * coeff_low
                            if val == 0: continue
                            j = 2 * I + x
                            k = 2 * J + y
                            V[j * p + k][c] += val

            for I in range(2):
                for K in range(2):
                    coeff_high = Sigma[I][K][q]
                    if coeff_high == 0: continue
                    for a in range(2):
                        for b in range(2):
                            coeff_low = Sigma[a][b][r]
                            if coeff_low == 0: continue
                            val = coeff_high * coeff_low
                            if val == 0: continue
                            i = 2 * I + a
                            k = 2 * K + b
                            W[k * n + i][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
