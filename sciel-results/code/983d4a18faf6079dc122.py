import numpy as np
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank_schoolbook = n * m * p
    U_s = np.zeros((n*m, rank_schoolbook))
    V_s = np.zeros((m*p, rank_schoolbook))
    W_s = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j, col] = 1.0
                V_s[j*p + k, col] = 1.0
                W_s[k*n + i, col] = 1.0
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
