"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-55 exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy: Split m=3 into m1=2, m2=1.
    - Part 1 (rank 20): A_right(2x1) * B_bottom(1x10) via schoolbook.
    - Part 2 (rank 35): A_left(2x2) * B_top(2x10) via 5 Strassen 2x2x2 blocks.
    
    Total rank = 20 + 35 = 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)
    
    col = 0
    
    # Part 1: A_right (A[:,2], shape 2x1) * B_bottom (B[2,:], shape 1x10)
    # A_right indices in U: A[0,2]=2, A[1,2]=5
    # B_bottom indices in V: B[2,k] = 2*10+k = 20+k for k=0..9
    # C[i,k] index in W: k*n+i = 2k+i
    for i in range(n):
        a_idx = i * m + 2  # 2 or 5
        for k in range(p):
            b_idx = 2 * p + k  # 20..29
            c_idx = k * n + i  # 2k+i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Part 2: A_left (A[:,:2], shape 2x2) * B_top (B[:2,:], shape 2x10)
    # A_left indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B_top indices in V: B[0,k]=k, B[1,k]=10+k for k=0..9
    # Split B_top into 5 blocks of 2 columns each.
    
    X00, X01, X10, X11 = 0, 1, 3, 4  # A_left element indices in U
    
    for j in range(5):
        # B block j covers columns 2j, 2j+1 of B_top
        Y00 = 2 * j          # B[0, 2j]
        Y01 = 2 * j + 1      # B[0, 2j+1]
        Y10 = 10 + 2 * j     # B[1, 2j]
        Y11 = 10 + 2 * j + 1 # B[1, 2j+1]
        
        # C block j covers columns 2j, 2j+1 of C
        # C[0, 2j] = W[4j], C[1, 2j] = W[4j+1]
        # C[0, 2j+1] = W[4j+2], C[1, 2j+1] = W[4j+3]
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        # Strassen's 7 products for 2x2 * 2x2:
        # Z = X * Y where X = [[X00,X01],[X10,X11]], Y = [[Y00,Y01],[Y10,Y11]]
        
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
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
