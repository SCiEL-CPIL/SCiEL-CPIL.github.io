import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for A_left (2x2) * B_top (2x10)
    # 5 blocks of 2x2
    for t in range(5):
        # B columns are 2t and 2t+1
        # V indices for B_top:
        # b00 is V[0*10 + 2t]
        # b01 is V[0*10 + 2t+1]
        # b10 is V[1*10 + 2t]
        # b11 is V[1*10 + 2t+1]
        
        v00 = 2 * t
        v01 = 2 * t + 1
        v10 = 10 + 2 * t
        v11 = 10 + 2 * t + 1
        
        # W indices for C block t:
        # c00 is W[2t*2 + 0] = W[4t]
        # c10 is W[2t*2 + 1] = W[4t+1]
        # c01 is W[(2t+1)*2 + 0] = W[4t+2]
        # c11 is W[(2t+1)*2 + 1] = W[4t+3]
        
        w00 = 4 * t
        w10 = 4 * t + 1
        w01 = 4 * t + 2
        w11 = 4 * t + 3
        
        # Factor 1: M1 = (a00 + a11)(b00 + b11)
        col = 7 * t
        U[0][col] += 1  # a00
        U[4][col] += 1  # a11
        V[v00][col] += 1 # b00
        V[v11][col] += 1 # b11
        W[w00][col] += 1 # c00
        W[w01][col] += 1 # c01
        W[w11][col] += 1 # c11
        
        # Factor 2: M2 = (a10 + a11)b00
        col = 7 * t + 1
        U[3][col] += 1  # a10
        U[4][col] += 1  # a11
        V[v00][col] += 1 # b00
        W[w10][col] += 1 # c10
        W[w11][col] += -1 # c11
        
        # Factor 3: M3 = a00(b01 - b11)
        col = 7 * t + 2
        U[0][col] += 1  # a00
        V[v01][col] += 1 # b01
        V[v11][col] += -1 # b11
        W[w01][col] += 1 # c01
        W[w11][col] += 1 # c11
        
        # Factor 4: M4 = a11(b10 - b00)
        col = 7 * t + 3
        U[4][col] += 1  # a11
        V[v10][col] += 1 # b10
        V[v00][col] += -1 # b00
        W[w00][col] += 1 # c00
        W[w10][col] += 1 # c10
        
        # Factor 5: M5 = (a00 + a01)b11
        col = 7 * t + 4
        U[0][col] += 1  # a00
        U[1][col] += 1  # a01
        V[v11][col] += 1 # b11
        W[w00][col] += -1 # c00
        W[w01][col] += 1 # c01
        
        # Factor 6: M6 = (a10 - a00)(b00 + b01)
        col = 7 * t + 5
        U[3][col] += 1  # a10
        U[0][col] += -1 # a00
        V[v00][col] += 1 # b00
        V[v01][col] += 1 # b01
        W[w11][col] += 1 # c11
        
        # Factor 7: M7 = (a01 - a11)(b10 + b11)
        col = 7 * t + 6
        U[1][col] += 1  # a01
        U[4][col] += -1 # a11
        V[v10][col] += 1 # b10
        V[v11][col] += 1 # b11
        W[w00][col] += 1 # c00

    # Schoolbook part for A_right (2x1) * B_bot (1x10)
    # A_right indices: U[2] (a02), U[5] (a12)
    # B_bot indices: V[20+k] (b2k)
    # Output C[i, k]: W[2k+i]
    
    for k in range(10):
        col = 35 + 2 * k
        U[2][col] = 1 # a02
        V[20 + k][col] = 1 # b2k
        W[2 * k][col] = 1 # c0k (i=0)
        
        col = 35 + 2 * k + 1
        U[5][col] = 1 # a12
        V[20 + k][col] = 1 # b2k
        W[2 * k + 1][col] = 1 # c1k (i=1)

    return {"U": U, "V": V, "W": W}
