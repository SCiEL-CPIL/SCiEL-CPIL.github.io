import numpy as np

def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 7-term Strassen decomposition for 2x2x2
    strassen_coeffs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    r = 0
    for b in range(6):
        for s in range(7):
            u, v, w = strassen_coeffs[s]
            for i_loc in range(2):
                for j_loc in range(2):
                    val = u[i_loc * 2 + j_loc]
                    if val != 0:
                        U[i_loc * m + j_loc][r] = val
                            
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v[j_loc * 2 + k_loc]
                    if val != 0:
                        V[j_loc * p + (2 * b + k_loc)][r] = val
                            
            for k_loc in range(2):
                for i_loc in range(2):
                    val = w[k_loc * 2 + i_loc]
                    if val != 0:
                        W[(2 * b + k_loc) * n + i_loc][r] = val
                            
            r += 1
    return {"U": U, "V": V, "W": W}
