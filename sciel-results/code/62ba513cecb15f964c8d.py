import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 4x4x4 (cols 0-3 of B)
    # Indices: i in 0..3, j in 0..3, k in 0..3
    
    # Inner Strassen coefficients for X and Y linear forms
    # Q_r = L_X * L_Y
    # L_X coeffs for x0,x1,x2,x3
    # L_Y coeffs for y0,y1,y2,y3
    
    inner_LX = [
        [1, 0, 0, 1], # Q0: x0+x3
        [0, 0, 1, 1], # Q1: x2+x3
        [1, 0, 0, 0], # Q2: x0
        [0, 0, 0, 1], # Q3: x3
        [1, 1, 0, 0], # Q4: x0+x1
        [1, 0, 1, 0], # Q5: x2-x0 -> wait, Q5 is (x2-x0). So [ -1, 0, 1, 0]
        [0, 1, 0, -1]  # Q6: x1-x3 -> [0, 1, 0, -1]
    ]
    # Correction for Q5: (x2 - x0) -> coeffs [-1, 0, 1, 0]
    inner_LX[5] = [-1, 0, 1, 0]
    
    inner_LY = [
        [1, 0, 0, 1], # Q0: y0+y3
        [1, 0, 0, 0], # Q1: y0
        [0, 0, 1, -1], # Q2: y2-y3
        [0, 1, 0, -1], # Q3: y1-y0 -> wait, Q3 is x3(y1-y0). So [0, 1, 0, -1]
        [0, 0, 0, 1], # Q4: y3
        [1, 1, 0, 0], # Q5: y0+y1
        [0, 0, 1, 1]  # Q6: y2+y3
    ]
    # Correction for Q3: (y1 - y0) -> coeffs [ -1, 1, 0, 0] ?
    # Q3 = x3 * (y1 - y0). So LY is y1 - y0. Coeffs: [-1, 1, 0, 0].
    inner_LY[3] = [-1, 1, 0, 0]
    
    # Reconstruction coefficients for Z from Q
    # z0 = Q0 - Q1 + Q3 + Q5
    # z1 = Q2 + Q4 + Q5 - Q6
    # z2 = Q1 - Q2 + Q3 + Q6
    # z3 = Q0 - Q1 + Q4 - Q6
    recon_inner = [
        [1, -1, 0, 1, 0, 1, 0], # z0
        [0, 0, 1, 0, 1, 1, -1], # z1
        [0, 1, -1, 1, 0, 0, 1], # z2
        [1, -1, 0, 0, 1, 0, -1]  # z3
    ]
    
    # Outer Strassen block definitions
    # X_s and Y_s are 2x2 matrices of blocks.
    # We define alpha[s][u][v][i][j] and beta[s][u][v][j][k]
    # i,j are 0..3 for A. j,k are 0..3 for B.
    
    # Helper to get block A[i', j'][u, v] -> A[2i'+u, 2j'+v]
    # We will build alpha and beta arrays.
    
    alpha = [[[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    beta = [[[[[0]*4 for _ in range(4)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0: X = A00+A11, Y = B00+B11
    for u in range(2):
        for v in range(2):
            alpha[0][u][v][0][v] += 1 # A[0, v] ? No. A00 is A[0:2, 0:2].
            # A[2i'+u, 2j'+v].
            # For A00: i'=0, j'=0. A[u, v].
            # For A11: i'=1, j'=1. A[2+u, 2+v].
            alpha[0][u][v][u][v] = 1
            alpha[0][u][v][2+u][2+v] = 1
            
            beta[0][u][v][u][v] = 1
            beta[0][u][v][2+u][2+v] = 1
            
    # s=1: X = A10+A11, Y = B00
    for u in range(2):
        for v in range(2):
            # A10: i'=1, j'=0. A[2+u, v]
            # A11: i'=1, j'=1. A[2+u, 2+v]
            alpha[1][u][v][2+u][v] = 1
            alpha[1][u][v][2+u][2+v] = 1
            
            # B00: j'=0, k'=0. B[u, v]
            beta[1][u][v][u][v] = 1
            
    # s=2: X = A00, Y = B10-B11
    for u in range(2):
        for v in range(2):
            # A00: A[u, v]
            alpha[2][u][v][u][v] = 1
            
            # B10: j'=1, k'=0. B[2+u, v]
            # B11: j'=1, k'=1. B[2+u, 2+v]
            beta[2][u][v][2+u][v] = 1
            beta[2][u][v][2+u][2+v] = -1
            
    # s=3: X = A11, Y = B01-B00
    for u in range(2):
        for v in range(2):
            # A11: A[2+u, 2+v]
            alpha[3][u][v][2+u][2+v] = 1
            
            # B01: j'=0, k'=1. B[u, 2+v]
            # B00: j'=0, k'=0. B[u, v]
            beta[3][u][v][u][2+v] = 1
            beta[3][u][v][u][v] = -1
            
    # s=4: X = A00+A01, Y = B11
    for u in range(2):
        for v in range(2):
            # A00: A[u, v]
            # A01: A[u, 2+v]
            alpha[4][u][v][u][v] = 1
            alpha[4][u][v][u][2+v] = 1
            
            # B11: B[2+u, 2+v]
            beta[4][u][v][2+u][2+v] = 1
            
    # s=5: X = A10-A00, Y = B00+B01
    for u in range(2):
        for v in range(2):
            # A10: A[2+u, v]
            # A00: A[u, v]
            alpha[5][u][v][2+u][v] = 1
            alpha[5][u][v][u][v] = -1
            
            # B00: B[u, v]
            # B01: B[u, 2+v]
            beta[5][u][v][u][v] = 1
            beta[5][u][v][u][2+v] = 1
            
    # s=6: X = A01-A11, Y = B10+B11
    for u in range(2):
        for v in range(2):
            # A01: A[u, 2+v]
            # A11: A[2+u, 2+v]
            alpha[6][u][v][u][2+v] = 1
            alpha[6][u][v][2+u][2+v] = -1
            
            # B10: B[2+u, v]
            # B11: B[2+u, 2+v]
            beta[6][u][v][2+u][v] = 1
            beta[6][u][v][2+u][2+v] = 1
            
    # Outer reconstruction coefficients
    # C[i'j'] = sum_s coeff[i'j', s] M_s
    # coeff[i'j', s]
    recon_outer = [
        [1, -1, 0, 1, 0, 1, 0], # C00 (i'=0, j'=0)
        [0, 0, 1, 0, 1, 1, -1], # C01 (i'=0, j'=1)
        [0, 1, -1, 1, 0, 0, 1], # C10 (i'=1, j'=0)
        [1, -1, 0, 0, 1, 0, -1]  # C11 (i'=1, j'=1)
    ]
    
    col = 0
    for s in range(7):
        for r in range(7):
            # Calculate U vector for this term
            # U[i, j] = sum_{u,v} inner_LX[r][u*2+v] * alpha[s][u][v][i][j]
            for i in range(4):
                for j in range(4):
                    val = 0
                    for u in range(2):
                        for v in range(2):
                            val += inner_LX[r][u*2+v] * alpha[s][u][v][i][j]
                    U[i*4+j][col] = val
            
            # Calculate V vector for this term
            # V[j, k] = sum_{u,v} inner_LY[r][u*2+v] * beta[s][u][v][j][k]
            for j in range(4):
                for k in range(4):
                    val = 0
                    for u in range(2):
                        for v in range(2):
                            val += inner_LY[r][u*2+v] * beta[s][u][v][j][k]
                    V[j*4+k][col] = val # Note: V is m x p. Here p=4 for this part.
                    # But V is m x 7. So index is j*7 + k.
                    # We are filling cols 0..3 of B.
                    # So V[j*7 + k][col] = val.
            
            # Calculate W vector for this term
            # W[k, i] = sum_{i',j',u,v} recon_outer[i'j'][s] * recon_inner[u*2+v][r] * (delta if i=2i'+u, k=2j'+v)
            # Actually, W[k, i] gets contribution if k=2j'+v and i=2i'+u.
            # Coeff is recon_outer[i'j'][s] * recon_inner[u*2+v][r]
            for i_prime in range(2):
                for j_prime in range(2):
                    for u in range(2):
                        for v in range(2):
                            i = 2*i_prime + u
                            k = 2*j_prime + v
                            val = recon_outer[i_prime*2+j_prime][s] * recon_inner[u*2+v][r]
                            W[k*4+i][col] += val # W is p x n. Here p=4, n=4.
                            # Full W is 7 x 4. Index k*4+i.
            
            col += 1
            
    # Part 2: Schoolbook for 4x4x3 (cols 4-6 of B)
    # i in 0..3, j in 0..3, k in 4..6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
