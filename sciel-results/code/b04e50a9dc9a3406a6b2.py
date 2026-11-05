import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 (Strassen) * 27 (Schoolbook 3x3) = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M_s = A_s * B_s
    # A_s = sum_{I', J'} alpha[s][I'][J'] * A[I', J']
    # B_s = sum_{J'', K'} beta[s][J''][K'] * B[J'', K']
    # C[I, K] = sum_s P[I][K][s] * M_s[I, K]
    
    alpha = [
        [[1, 0], [0, 1]],  # M0: A00 + A11
        [[0, 0], [1, 1]],  # M1: A10 + A11
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: A00 + A01
        [[-1, 0], [1, 0]], # M5: A10 - A00
        [[0, 1], [0, -1]]  # M6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # M0: B00 + B11
        [[1, 0], [0, 0]],  # M1: B00
        [[0, 1], [0, -1]], # M2: B01 - B11
        [[-1, 0], [0, 1]], # M3: B10 - B00
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: B00 + B01
        [[0, 1], [1, 1]]   # M6: B10 + B11
    ]
    
    P = [
        [[1, 1], [1, 1]],  # C00, C01, C10, C11 for M0
        [[0, 1], [1, 0]],  # M1
        [[0, 1], [0, 1]],  # M2
        [[1, 0], [1, -1]], # M3
        [[0, 1], [0, 1]],  # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 0], [0, -1]]  # M6
    ]
    # Note: P[I][K][s] is the coefficient for C[I,K] from M_s
    # C00 = M0 + M3 - M4 + M6 -> P[0][0][0]=1, P[0][0][3]=1, P[0][0][4]=-1, P[0][0][6]=1
    # C01 = M2 + M4 -> P[0][1][2]=1, P[0][1][4]=1
    # C10 = M1 + M3 -> P[1][0][1]=1, P[1][0][3]=1
    # C11 = M0 - M1 + M2 + M5 -> P[1][1][0]=1, P[1][1][1]=-1, P[1][1][2]=1, P[1][1][5]=1
    
    # Correcting P based on standard Strassen:
    # C00 = M1 - M2 + M4 - M7 (using 1-based)
    # Let's use the 0-based formulas derived above.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    P_corrected = [
        [[1, 1], [1, 1]], # s=0
        [[0, 1], [1, 0]], # s=1
        [[0, 1], [0, 1]], # s=2
        [[1, 0], [1, -1]], # s=3
        [[0, 1], [0, 1]], # s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 0], [0, -1]] # s=6
    ]
    # Wait, P[I][K][s]
    # P[0][0][0]=1, P[0][1][0]=1, P[1][0][0]=1, P[1][1][0]=1
    # P[0][0][1]=0, P[0][1][1]=1, P[1][0][1]=1, P[1][1][1]=0
    # P[0][0][2]=0, P[0][1][2]=1, P[1][0][2]=0, P[1][1][2]=1
    # P[0][0][3]=1, P[0][1][3]=0, P[1][0][3]=1, P[1][1][3]=-1
    # P[0][0][4]=0, P[0][1][4]=1, P[1][0][4]=0, P[1][1][4]=1
    # P[0][0][5]=0, P[0][1][5]=0, P[1][0][5]=0, P[1][1][5]=1
    # P[0][0][6]=1, P[0][1][6]=0, P[1][0][6]=0, P[1][1][6]=-1
    
    # My P_corrected is indexed [s][I][K]
    # I need [I][K][s] or just access P[s][I][K]
    
    for s in range(7):
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    r = s * 27 + ip * 9 + jp * 3 + kp
                    # U
                    for I in range(2):
                        for J in range(2):
                            coeff = alpha[s][I][J]
                            if coeff != 0:
                                u_idx = (I * 3 + ip) * 6 + (J * 3 + jp)
                                U[u_idx][r] += coeff
                    # V
                    for J2 in range(2):
                        for K in range(2):
                            coeff = beta[s][J2][K]
                            if coeff != 0:
                                v_idx = (J2 * 3 + jp) * 6 + (K * 3 + kp)
                                V[v_idx][r] += coeff
                    # W
                    for I in range(2):
                        for K in range(2):
                            coeff = P_corrected[s][I][K]
                            if coeff != 0:
                                w_idx = (K * 3 + kp) * 6 + (I * 3 + ip)
                                W[w_idx][r] += coeff
                                
    return {"U": U, "V": V, "W": W}
