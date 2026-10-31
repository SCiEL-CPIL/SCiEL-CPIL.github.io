import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen factors
    # U2[i, j, c], V2[j, k, c], W2[k, i, c]
    # c in 0..6
    U2 = [[[0]*2 for _ in range(2)] for _ in range(7)]
    V2 = [[[0]*2 for _ in range(2)] for _ in range(7)]
    W2 = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # c=0 (M1)
    U2[0][0][0] = 1; U2[0][1][1] = 1
    V2[0][0][0] = 1; V2[0][1][1] = 1
    W2[0][0][0] = 1; W2[0][1][1] = -1
    
    # c=1 (M2)
    U2[1][1][0] = 1; U2[1][1][1] = 1
    V2[1][0][0] = 1
    W2[1][1][0] = 1; W2[1][1][1] = -1
    
    # c=2 (M3)
    U2[2][0][0] = 1
    V2[2][0][1] = 1; V2[2][1][1] = -1
    W2[2][0][1] = 1; W2[2][1][1] = 1
    
    # c=3 (M4)
    U2[3][1][1] = 1
    V2[3][1][0] = 1; V2[3][0][0] = -1
    W2[3][0][0] = 1; W2[3][1][0] = 1
    
    # c=4 (M5)
    U2[4][0][0] = 1; U2[4][0][1] = 1
    V2[4][1][1] = 1
    W2[4][0][0] = -1; W2[4][0][1] = 1
    
    # c=5 (M6)
    U2[5][1][0] = 1; U2[5][0][0] = -1
    V2[5][0][0] = 1; V2[5][0][1] = 1
    W2[5][1][1] = 1
    
    # c=6 (M7)
    U2[6][0][1] = 1; U2[6][1][1] = -1
    V2[6][1][0] = 1; V2[6][1][1] = 1
    W2[6][0][0] = 1; W2[6][1][1] = 1

    # Construct full factors
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            
            # U
            for I in range(2):
                for ip in range(2):
                    for J in range(2):
                        for jp in range(2):
                            val = U2[c1][I][J] * U2[c2][ip][jp]
                            if val != 0:
                                U[2*I + ip][2*J + jp][c] += val
                                
            # V
            for J in range(2):
                for jp in range(2):
                    for K in range(2):
                        for kp in range(2):
                            val = V2[c1][J][K] * V2[c2][jp][kp]
                            if val != 0:
                                V[2*J + jp][2*K + kp][c] += val
                                
            # W
            for K in range(2):
                for kp in range(2):
                    for I in range(2):
                        for ip in range(2):
                            val = W2[c1][K][I] * W2[c2][kp][ip]
                            if val != 0:
                                W[2*K + kp][2*I + ip][c] += val

    return {"U": U, "V": V, "W": W}
