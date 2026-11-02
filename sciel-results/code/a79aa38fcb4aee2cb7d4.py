import numpy as np
import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # Use rank 42 decomposition (6 Strassen blocks)
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U_forms for Strassen 2x2x2
    # Each form is [a00, a01, a10, a11]
    U_forms = [
        [1, 0, 0, 1],   # S0: a00 + a11
        [0, 0, 1, 1],   # S1: a10 + a11
        [1, 0, 0, 0],   # S2: a00
        [0, 0, 0, 1],   # S3: a11
        [1, 1, 0, 0],   # S4: a00 + a01
        [-1, 0, 1, 0],  # S5: a10 - a00
        [0, 1, 0, -1]   # S6: a01 - a11
    ]
    
    for u in range(p // 2):
        for t in range(7):
            c = u * 7 + t
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
