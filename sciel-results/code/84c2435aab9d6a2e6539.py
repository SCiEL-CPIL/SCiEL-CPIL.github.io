import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrix multiplication
    # U: Left matrix coefficients
    # V: Right matrix coefficients
    # W: Result matrix coefficients
    # Format: list of lists of tuples (coeff, row, col)
    
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (1, 1, 1)],       # m2
        [(1, 0, 0)],                  # m3
        [(1, 1, 1)],                  # m4
        [(1, 0, 0), (1, 0, 1)],       # m5
        [(1, 1, 0), (-1, 0, 0)],      # m6
        [(1, 0, 1), (-1, 1, 1)],      # m7
    ]
    
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 0, 0)],                  # m2
        [(1, 0, 1), (-1, 1, 1)],      # m3
        [(1, 1, 0), (-1, 0, 0)],      # m4
        [(1, 1, 1)],                  # m5
        [(1, 0, 0), (1, 0, 1)],       # m6
        [(1, 1, 0), (1, 1, 1)],       # m7
    ]
    
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (-1, 1, 1)],      # m2
        [(1, 0, 1), (1, 1, 1)],       # m3
        [(1, 0, 0), (1, 1, 0)],       # m4
        [(-1, 0, 0), (1, 0, 1)],      # m5
        [(1, 1, 1)],                  # m6
        [(1, 0, 0)],                  # m7
    ]
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for outer_s in range(7):
        for inner_s in range(7):
            # U
            for (oc, obr, obc) in strassen_U[outer_s]:
                for (ic, ibr, ibc) in strassen_U[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    U[gr * m + gc, col] += oc * ic
            
            # V
            for (oc, obr, obc) in strassen_V[outer_s]:
                for (ic, ibr, ibc) in strassen_V[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    V[gr * p + gc, col] += oc * ic
            
            # W
            for (oc, obr, obc) in strassen_W[outer_s]:
                for (ic, ibr, ibc) in strassen_W[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    W[gc * n + gr, col] += oc * ic
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
