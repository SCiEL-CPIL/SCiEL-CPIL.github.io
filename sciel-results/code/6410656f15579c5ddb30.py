"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 145.0

# EVOLVE-BLOCK-START
def construct():
    """
    Use a 2x2x2 block decomposition with Strassen's algorithm for each
    2x2 matrix multiplication. This yields a valid rank-189 decomposition
    for 6x6x6 matrix multiplication (3*3*3 = 27 block-mults * 7 = 189).
    """
    n, m, p = DIMENSIONS

    # Verified Strassen decomposition for 2x2 matrix multiplication.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
    # C00 = ae+bg, C01 = af+bh, C10 = ce+dg, C11 = cf+dh
    # 7 products with verified W coefficients:
    strassen = [
        # M0 = (A00+A11)(B00+B11): C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M1 = (A10+A11)B00: C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M2 = A00(B01-B11): C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M3 = A11(B10-B00): C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M4 = (A00+A01)B11: C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M5 = (A10-A00)(B00+B01): C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M6 = (A01-A11)(B10+B11): C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    block = 2
    nb = n // block
    mb = m // block
    pb = p // block
    total_rank = nb * mb * pb * len(strassen)  # 3*3*3*7 = 189

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    col = 0
    for I in range(nb):
        for J in range(mb):
            for K in range(pb):
                for u_c, v_c, w_c in strassen:
                    for (ri, ci, val) in u_c:
                        U[(I * block + ri) * m + (J * block + ci), col] += val
                    for (ri, ci, val) in v_c:
                        V[(J * block + ri) * p + (K * block + ci), col] += val
                    for (ri, ci, val) in w_c:
                        W[(K * block + ci) * n + (I * block + ri), col] += val
                    col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
