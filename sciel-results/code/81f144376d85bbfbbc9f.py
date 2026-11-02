# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nb = 3
    bs = 2
    
    rank = nb * nb * nb * 7
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen_terms = [
        (
            [((0,0), 1), ((1,1), 1)], 
            [((0,0), 1), ((1,1), 1)], 
            [((0,0), 1), ((1,1), 1)]
        ),
        (
            [((1,0), 1), ((1,1), 1)], 
            [((0,0), 1)], 
            [((1,0), 1), ((1,1), -1)]
        ),
        (
            [((0,0), 1)], 
            [((0,1), 1), ((1,1), -1)], 
            [((0,1), 1), ((1,1), 1)]
        ),
        (
            [((1,1), 1)], 
            [((1,0), 1), ((0,0), -1)], 
            [((0,0), 1), ((1,0), 1)]
        ),
        (
            [((0,0), 1), ((0,1), 1)], 
            [((1,1), 1)], 
            [((0,0), -1), ((0,1), 1)]
        ),
        (
            [((1,0), 1), ((0,0), -1)], 
            [((0,0), 1), ((0,1), 1)], 
            [((1,1), 1)]
        ),
        (
            [((0,1), 1), ((1,1), -1)], 
            [((1,0), 1), ((1,1), 1)], 
            [((0,0), 1)]
        ),
    ]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                for x_factors, y_factors, c_factors in strassen_terms:
                    for (lr, lc), val in x_factors:
                        r = i_block * bs + lr
                        c = j_block * bs + lc
                        U[r * m + c, col] += val
                    
                    for (lr, lc), val in y_factors:
                        r = j_block * bs + lr
                        c = k_block * bs + lc
                        V[r * p + c, col] += val
                    
                    for (lr, lc), val in c_factors:
                        r = i_block * bs + lr
                        c = k_block * bs + lc
                        W[c * n + r, col] += val
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
