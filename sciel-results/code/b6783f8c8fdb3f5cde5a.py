import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 coefficients
    # U_r, V_r, W_r are 2x2 matrices
    # r=0..6
    U2 = []
    V2 = []
    W2 = []
    
    # M1 = (A00+A11)(B00+B11)
    U2.append([[1, 0], [0, 1]])
    V2.append([[1, 0], [0, 1]])
    W2.append([[1, 0], [0, 1]]) # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U2.append([[0, 0], [1, 1]])
    V2.append([[1, 0], [0, 0]])
    W2.append([[0, 0], [1, 0]]) # C10 += M2
    
    # M3 = A00(B01-B11)
    U2.append([[1, 0], [0, 0]])
    V2.append([[0, 1], [0, -1]])
    W2.append([[0, 1], [0, 0]]) # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U2.append([[0, 0], [0, 1]])
    V2.append([[0, 0], [1, -1]])
    W2.append([[1, 0], [0, 0]]) # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U2.append([[1, 1], [0, 0]])
    V2.append([[0, 0], [0, 1]])
    W2.append([[-1, 0], [0, 1]]) # C00 -= M5, C01 += M5, C11 += M6? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 contributes to C00 (-1) and C01 (+1).
    W2[4] = [[-1, 1], [0, 0]]
    
    # M6 = (A10-A00)(B00+B01)
    U2.append([[1, 0], [-1, 0]])
    V2.append([[1, 1], [0, 0]])
    W2.append([[0, 0], [0, 1]]) # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U2.append([[0, 1], [0, -1]])
    V2.append([[0, 0], [1, 1]])
    W2.append([[1, 0], [0, 0]]) # C00 += M7
    
    # Correcting W2 based on formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W2 = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 0], [1, 0]], # M2 -> C10
        [[0, 1], [0, 0]], # M3 -> C01, C11 (Wait, C11 has M3)
        [[1, 0], [0, 0]], # M4 -> C00, C10
        [[-1, 1], [0, 0]],# M5 -> C00, C01
        [[0, 0], [0, 1]], # M6 -> C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    # Fix M3 W: C11 = M1 - M2 + M3 + M6. So M3 contributes to C01 and C11.
    W2[2] = [[0, 1], [0, 1]]
    
    # Schoolbook 3x3
    # 27 terms
    U3 = []
    V3 = []
    W3 = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                u = [[0]*3 for _ in range(3)]
                v = [[0]*3 for _ in range(3)]
                w = [[0]*3 for _ in range(3)]
                u[x][y] = 1
                v[y][z] = 1
                w[z][x] = 1 # W[k*n+i] -> w[k'][i']
                U3.append(u)
                V3.append(v)
                W3.append(w)
                
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for r in range(7):
        for s in range(27):
            u2 = U2[r]
            v2 = V2[r]
            w2 = W2[r]
            u3 = U3[s]
            v3 = V3[s]
            w3 = W3[s]
            
            for a in range(2):
                for b in range(2):
                    if u2[a][b] != 0:
                        for i_prime in range(3):
                            for j_prime in range(3):
                                if u3[i_prime][j_prime] != 0:
                                    i = 3*a + i_prime
                                    j = 3*b + j_prime
                                    U[i*6+j][col] = u2[a][b] * u3[i_prime][j_prime]
                                    
            for c in range(2):
                for d in range(2):
                    if v2[c][d] != 0:
                        for j_prime in range(3):
                            for k_prime in range(3):
                                if v3[j_prime][k_prime] != 0:
                                    j = 3*c + j_prime
                                    k = 3*d + k_prime
                                    V[j*6+k][col] = v2[c][d] * v3[j_prime][k_prime]
                                    
            for f in range(2):
                for e in range(2):
                    if w2[f][e] != 0:
                        for k_prime in range(3):
                            for i_prime in range(3):
                                if w3[k_prime][i_prime] != 0:
                                    k = 3*e + k_prime
                                    i = 3*f + i_prime
                                    W[k*6+i][col] = w2[f][e] * w3[k_prime][i_prime]
                                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
