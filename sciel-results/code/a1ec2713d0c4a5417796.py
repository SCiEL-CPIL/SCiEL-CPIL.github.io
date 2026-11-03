"""
Rewritten program for matrix multiplication 2x3x10.
Uses a Strassen-based decomposition for the 2x2x10 sub-problem and a direct
decomposition for the 2x1x10 sub-problem.
Total rank: 35 + 20 = 55.
"""
import time
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
       - Part 1: (2x2) * (2x10) -> (2x10)
       - Part 2: (2x1) * (1x10) -> (2x10)
    3. Part 2 (2x1x10) requires rank 20 (outer product of 2x1 and 1x10).
    4. Part 1 (2x2x10) is solved by splitting the 10 columns into 5 blocks of 2.
       Each 2x2x2 block is solved using Strassen's algorithm (rank 7).
       Total rank for Part 1: 5 * 7 = 35.
    5. Total rank = 20 + 35 = 55.
    
    Returns:
    dict with keys 'U', 'V', 'W' containing lists of lists.
    U: shape (6, 55)
    V: shape (30, 55)
    W: shape (20, 55)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_right (2x1) * B_bottom (1x10)
    # A is 2x3. A_right is columns 2 (index 2).
    # A[0,2] is U index 2. A[1,2] is U index 5.
    # B is 3x10. B_bottom is row 2 (index 2).
    # B[2,k] is V index 2*10 + k = 20+k.
    # C is 2x10. C[i,k] is W index k*2 + i.
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
            
    # Part 2: A_left (2x2) * B_top (2x10)
    # A_left is columns 0,1.
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4.
    # B_top is rows 0,1.
    # B[0,k]=k, B[1,k]=10+k.
    
    # We process 5 blocks of 2 columns from B_top.
    # Block j covers columns 2j and 2j+1.
    
    X00, X01, X10, X11 = 0, 1, 3, 4  # Indices in U for A_left
    
    for j in range(5):
        if time.time() > deadline:
            break
            
        # Indices in V for B_top block j
        Y00 = 2 * j       # B[0, 2j]
        Y01 = 2 * j + 1   # B[0, 2j+1]
        Y10 = 10 + 2 * j  # B[1, 2j]
        Y11 = 10 + 2 * j + 1 # B[1, 2j+1]
        
        # Indices in W for C block j
        # C[0, 2j] = 4j
        # C[1, 2j] = 4j+1
        # C[0, 2j+1] = 4j+2
        # C[1, 2j+1] = 4j+3
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        # Strassen's Algorithm for 2x2 * 2x2
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
        
    # If we hit the deadline and didn't finish, we might have an incomplete matrix.
    # However, the loop is small (5 iterations), so it should finish instantly.
    # The deadline guard is just for safety against unexpected slowness.
    
    if col != 55:
        # Fallback to schoolbook if something went wrong
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
