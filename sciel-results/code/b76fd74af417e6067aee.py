import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 120s limit, keep safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split m=3 into m1=2 and m2=1.
    # T_{2,3,8} = T_{2,2,8} + T_{2,1,8}
    # T_{2,2,8} uses Strassen on 2x2x2 blocks. 4 blocks * 7 rank = 28.
    # T_{2,1,8} uses schoolbook. 2 * 1 * 8 = 16 rank.
    # Total rank = 44.
    
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # Indices:
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # 1. (a+d)(e+h)
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 2. (c+d)e
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        # 3. a(f-h)
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 4. d(g-e)
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # 5. (a+b)h
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        # 6. (c-a)(e+f)
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 7. (b-d)(g+h)
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off), col] = 1
                V[(j_off)*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # Block 1: k=0..3
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 0) # j=0,1 k=0,1
    add_strassen_2x2x2(0, 0, 2) # j=0,1 k=2,3
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 0) # j=2 k=0..3

    # Block 2: k=4..7
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 4) # j=0,1 k=4,5
    add_strassen_2x2x2(0, 0, 6) # j=0,1 k=6,7
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 4) # j=2 k=4..7

    if time.time() > deadline:
        # This should never happen for this fixed construction
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
