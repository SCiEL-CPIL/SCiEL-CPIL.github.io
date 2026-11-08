import numpy as np

def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: 1x4x8 (Rank 32)
    for j in range(4):
        for k in range(8):
            U[0 * 4 + j][r] = 1
            V[j * 8 + k][r] = 1
            W[k * 3 + 0][r] = 1
            r += 1
            
    # Part 2: 2x4x8 (Rank 56)
    terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
        ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,-1],[0,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,1],[1,1]], [[1,0],[0,1]])
    ]
    
    for b in range(2):
        for q in range(4):
            for t in terms:
                U_t, V_t, W_t = t
                for i in range(2):
                    for jp in range(2):
                        val = U_t[i][jp]
                        if val != 0:
                            U[(1 + i) * 4 + (2 * b + jp)][r] = val
                            
                for jp in range(2):
                    for kp in range(2):
                        val = V_t[jp][kp]
                        if val != 0:
                            V[(2 * b + jp) * 8 + (kp * 4 + q)][r] = val
                            
                for kp in range(2):
                    for i in range(2):
                        val = W_t[kp][i]
                        if val != 0:
                            W[((kp * 4 + q) * 3) + (1 + i)][r] = val
                            
                r += 1
                
    return {"U": U, "V": V, "W": W}
