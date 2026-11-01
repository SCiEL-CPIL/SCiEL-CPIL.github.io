import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    col = 0
    for jm in range(2):
        for kp in range(3):
            for r_idx in range(7):
                r = col + r_idx
                A_signs, B_signs, C_signs = strassen_defs[r_idx]
                
                for loc_idx, sign in A_signs.items():
                    li, lj = divmod(loc_idx, 2)
                    a = li * m + (jm * 2 + lj)
                    U[a, r] = sign
                    
                for loc_idx, sign in B_signs.items():
                    lj, lk = divmod(loc_idx, 2)
                    b = (jm * 2 + lj) * p + (kp * 2 + lk)
                    V[b, r] = sign
                    
                for loc_idx, sign in C_signs.items():
                    li, ll = divmod(loc_idx, 2)
                    c = (kp * 2 + ll) * n + li
                    W[c, r] = sign
                    
            col += 7
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
