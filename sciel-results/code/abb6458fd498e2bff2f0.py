"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix.
    
    Strategy:
    1. Split the middle dimension m=3 into m1=2 and m2=1.
    2. The problem decomposes into:
       - Part 1: (2x2) * (2x10) -> (2x10) using Strassen per 2-column block
       - Part 2: (2x1) * (1x10) -> (2x10) as outer product
    3. Part 2 requires 2*1*10 = 20 multiplications.
    4. Part 1 uses 5 blocks of 2x2 * 2x2, each with Strassen (7 mults) = 35 mults.
    Total rank = 55.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_right (2x1) * B_bottom (1x10)
    # A_right is A[:, 2]. Indices in U: 2, 5.
    # B_bottom is B[2, :]. Indices in V: 20..29.
    # C[:, k] gets A[i, 2] * B[2, k].
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    # Part 2: A_left (2x2) * B_top (2x10) using Strassen blocks
    # A_left is A[:, 0:2]. Indices in U: 0,1 (row 0), 3,4 (row 1).
    # B_top is B[0:2, :]. Indices in V: 0..9 (row 0), 10..19 (row 1).
    
    # Strassen indices for X (A_left)
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        # B block columns: 2j, 2j+1
        # Y00 = B[0, 2j] -> V[2j]
        # Y01 = B[0, 2j+1] -> V[2j+1]
        # Y10 = B[1, 2j] -> V[10+2j]
        # Y11 = B[1, 2j+1] -> V[10+2j+1]
        
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        # C block columns: 2j, 2j+1
        # Z00 = C[0, 2j] -> W[4j]
        # Z01 = C[0, 2j+1] -> W[4j+2]
        # Z10 = C[1, 2j] -> W[4j+1]
        # Z11 = C[1, 2j+1] -> W[4j+3]
        
        Z00 = 4 * j
        Z01 = 4 * j + 2
        Z10 = 4 * j + 1
        Z11 = 4 * j + 3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[X00, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[X10, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1
        W[Z11, col] -= 1
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[X00, col] += 1
        V[Y01, col] += 1
        V[Y11, col] -= 1
        W[Z01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[X11, col] += 1
        V[Y10, col] += 1
        V[Y00, col] -= 1
        W[Z00, col] += 1
        W[Z10, col] += 1
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[X00, col] += 1
        U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1
        W[Z01, col] += 1
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[X10, col] += 1
        U[X00, col] -= 1
        V[Y00, col] += 1
        V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[X01, col] += 1
        U[X11, col] -= 1
        V[Y10, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
        
    # Verify col == 55
    if col != 55:
        # Fallback to schoolbook if logic error
        rank = 60
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
