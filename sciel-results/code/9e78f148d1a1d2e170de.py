import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Block dimensions
    # A: 2x4 -> 2 blocks of 1x2. (i, a) where a in {0,1}
    # B: 4x8 -> 2x2 blocks of 2x4. (a, c) where a in {0,1}, c in {0,1}
    # C: 2x8 -> 2 blocks of 1x4. (i, c) where c in {0,1}
    
    # Strassen rank 7
    # Block schoolbook rank 8 (1x2 by 2x4)
    rank = 7 * 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # SA[i][a], SB[a][c], SC[i][c] for each step s
    # s=0..6
    
    # M1
    SA0 = [[1, 0], [0, 1]] # A00, A11
    SB0 = [[1, 1], [1, 1]] # B00, B01, B10, B11? No.
    # M1 = (A00 + A11)(B00 + B11)
    # A blocks: (0,0), (1,1)
    # B blocks: (0,0), (1,1)
    SA[0][0][0] = 1; SA[0][1][1] = 1
    SB[0][0][0] = 1; SB[0][1][1] = 1
    # C blocks for M1:
    # C00 += M1, C11 += M1
    SC[0][0][0] = 1; SC[0][1][1] = 1
    
    # M2 = (A10 + A11) B00
    SA[1][1][0] = 1; SA[1][1][1] = 1
    SB[1][0][0] = 1
    # C10 += M2
    SC[1][1][0] = 1
    
    # M3 = A00 (B11 - B10)
    SA[2][0][0] = 1
    SB[2][1][1] = 1; SB[2][1][0] = -1
    # C01 += M3
    SC[2][0][1] = 1
    
    # M4 = A11 (B00 - B01)
    SA[3][1][1] = 1
    SB[3][0][0] = 1; SB[3][0][1] = -1
    # C00 += M4, C10 += M4
    SC[3][0][0] = 1; SC[3][1][0] = 1
    
    # M5 = (A00 + A01) B11
    SA[4][0][0] = 1; SA[4][0][1] = 1
    SB[4][1][1] = 1
    # C00 -= M5, C01 += M5
    SC[4][0][0] = -1; SC[4][0][1] = 1
    
    # M6 = (A10 - A00) (B00 + B01)
    SA[5][1][0] = 1; SA[5][0][0] = -1
    SB[5][0][0] = 1; SB[5][0][1] = 1
    # C11 += M6
    SC[5][1][1] = 1
    
    # M7 = (A01 - A11) (B10 + B11)
    SA[6][0][1] = 1; SA[6][1][1] = -1
    SB[6][1][0] = 1; SB[6][1][1] = 1
    # C00 += M7
    SC[6][0][0] = 1
    
    # Fill U, V, W
    for s in range(7):
        for t in range(8):
            b = t // 4
            d = t % 4
            r = 8 * s + t
            
            # U
            for i in range(2):
                for a in range(2):
                    coeff = SA[s][i][a]
                    if coeff != 0:
                        # Global j = 2a + b
                        j = 2 * a + b
                        U[i * m + j][r] += coeff
                        # Note: U_block is 1 for this b.
                        # If b != t//4, it would be 0, but we iterate t so b matches.
                        
            # V
            for a in range(2):
                for c in range(2):
                    coeff = SB[s][a][c]
                    if coeff != 0:
                        # Global j = 2a + b
                        # Global k = 4c + d
                        j = 2 * a + b
                        k = 4 * c + d
                        V[j * p + k][r] += coeff
                        # V_block is 1 for this (b,d).
                        
            # W
            for i in range(2):
                for c in range(2):
                    coeff = SC[s][i][c]
                    if coeff != 0:
                        # Global k = 4c + d
                        k = 4 * c + d
                        W[k * n + i][r] += coeff
                        # W_block is 1 for this d.
                        
    return {"U": U, "V": V, "W": W}
