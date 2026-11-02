import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 2, 6
    
    # --- Rank 21 Solution (Strassen) ---
    rank_21 = 21
    U21 = np.zeros((n * m, rank_21))
    V21 = np.zeros((m * p, rank_21))
    W21 = np.zeros((p * n, rank_21))
    
    alpha = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    beta = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    gamma = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]

    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            for (i, j), val in alpha[s].items():
                U21[i * m + j, r] = val
            for (j, k_local), val in beta[s].items():
                k_global = b * 2 + k_local
                V21[j * p + k_global, r] = val
            for (k_local, i), val in gamma[s].items():
                k_global = b * 2 + k_local
                W21[k_global * n + i, r] = val

    return {
        "U": U21.tolist(),
        "V": V21.tolist(),
        "W": W21.tolist()
    }
