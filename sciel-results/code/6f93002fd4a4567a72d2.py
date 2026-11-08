import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_2 = np.zeros((7, 2, 2), dtype=int)
    V_2 = np.zeros((7, 2, 2), dtype=int)
    W_2 = np.zeros((7, 2, 2), dtype=int)
    
    U_2[0, 0, 0] = 1; U_2[0, 1, 1] = 1
    V_2[0, 0, 0] = 1; V_2[0, 1, 1] = 1
    W_2[0, 0, 0] = 1; W_2[0, 1, 1] = 1
    
    U_2[1, 1, 0] = 1; U_2[1, 1, 1] = 1
    V_2[1, 0, 0] = 1
    W_2[1, 1, 0] = 1; W_2[1, 1, 1] = -1
    
    U_2[2, 0, 0] = 1
    V_2[2, 0, 1] = 1; V_2[2, 1, 1] = -1
    W_2[2, 0, 1] = 1; W_2[2, 1, 1] = 1
    
    U_2[3, 1, 1] = 1
    V_2[3, 1, 0] = 1; V_2[3, 0, 0] = -1
    W_2[3, 0, 0] = 1; W_2[3, 1, 0] = 1
    
    U_2[4, 0, 0] = 1; U_2[4, 0, 1] = 1
    V_2[4, 1, 1] = 1
    W_2[4, 0, 0] = -1; W_2[4, 0, 1] = 1
    
    U_2[5, 1, 0] = 1; U_2[5, 0, 0] = -1
    V_2[5, 0, 0] = 1; V_2[5, 0, 1] = 1
    W_2[5, 1, 1] = 1
    
    U_2[6, 0, 1] = 1; U_2[6, 1, 1] = -1
    V_2[6, 1, 0] = 1; V_2[6, 1, 1] = 1
    W_2[6, 0, 0] = 1; W_2[6, 1, 0] = 1
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ui = i % 2
                    uj = j % 2
                    U[i * 4 + j][r] = U_2[s, ui, uj] * U_2[t, bi, bj]
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    vj = j % 2
                    vk = k % 2
                    V[j * 4 + k][r] = V_2[s, vj, vk] * V_2[t, bj, bk]
            for k in range(4):
                for i in range(4):
                    bk = k // 2
                    bi = i // 2
                    wk = k % 2
                    wi = i % 2
                    W[k * 4 + i][r] = W_2[s, wk, wi] * W_2[t, bk, bi]
                    
    return {"U": U, "V": V, "W": W}
