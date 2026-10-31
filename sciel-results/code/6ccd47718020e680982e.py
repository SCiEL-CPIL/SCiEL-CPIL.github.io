"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Decompose 4x4 matrix multiplication using an explicit enumeration of the
    49 terms from the recursive Strassen algorithm.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Base Strassen 2x2 factors
    # Format: list of 7 entries, each a list of (coeff, row, col)
    sU = [
        [(1, 0, 0), (1, 1, 1)],  # M1: A00 + A11
        [(1, 1, 0), (1, 1, 1)],  # M2: A10 + A11
        [(1, 0, 0)],             # M3: A00
        [(1, 1, 1)],             # M4: A11
        [(1, 0, 0), (1, 0, 1)],  # M5: A00 + A01
        [(1, 1, 0), (-1, 0, 0)], # M6: A10 - A00
        [(1, 0, 1), (-1, 1, 1)], # M7: A01 - A11
    ]
    sV = [
        [(1, 0, 0), (1, 1, 1)],  # M1: B00 + B11
        [(1, 0, 0)],             # M2: B00
        [(1, 0, 1), (-1, 1, 1)], # M3: B01 - B11
        [(1, 1, 0), (-1, 0, 0)], # M4: B10 - B00
        [(1, 1, 1)],             # M5: B11
        [(1, 0, 0), (1, 0, 1)],  # M6: B00 + B01
        [(1, 1, 0), (1, 1, 1)],  # M7: B10 + B11
    ]
    sW = [
        [(1, 0, 0), (1, 1, 1)],  # M1: C00 + C11
        [(1, 1, 0), (-1, 1, 1)], # M2: C10 - C11
        [(1, 0, 1), (1, 1, 1)],  # M3: C01 + C11
        [(1, 0, 0), (1, 1, 0)],  # M4: C00 + C10
        [(-1, 0, 0), (1, 0, 1)], # M5: -C00 + C01
        [(1, 1, 1)],             # M6: C11
        [(1, 0, 0)],             # M7: C00
    ]
    
    # Outer Strassen factors
    oU = sU
    oV = sV
    oW = sW
    
    col = 0
    for o in range(7):
        for i in range(7):
            # Fill U
            for (oc, obr, obc) in oU[o]:
                for (ic, ir, icl) in sU[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    U[gr * m + gc, col] += oc * ic
            
            # Fill V
            for (oc, obr, obc) in oV[o]:
                for (ic, ir, icl) in sV[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    V[gr * p + gc, col] += oc * ic
            
            # Fill W
            # W index = k * n + i
            for (oc, obr, obc) in oW[o]:
                for (ic, ir, icl) in sW[i]:
                    gi = 2 * obr + ir
                    gk = 2 * obc + icl
                    W[gk * n + gi, col] += oc * ic
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
