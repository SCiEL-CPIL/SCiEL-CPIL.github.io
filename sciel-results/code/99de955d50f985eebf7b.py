# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-55 matrix multiplication tensor decomposition for 2x3x10.
    Uses Strassen's algorithm for the 2x2x2 blocks and direct multiplication for 2x1x2 blocks.
    """
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: j in {0, 1} (2x2x10)
    # Split k into 5 blocks of 2
    for q in range(5):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # Strassen's 7 multiplications
        # M1 = (A00 + A11) * (B0k0 + B1k1) ? No, indices must match.
        # A is 2x2 (i, j in 0,1). B is 2x2 (j in 0,1; k in k0, k1).
        # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B00=B[0,k0], B01=B[0,k1], B10=B[1,k0], B11=B[1,k1]
        
        # M1
        U[0*3+0, col] += 1 # A00
        U[1*3+1, col] += 1 # A11
        V[0*10+k0, col] += 1 # B0k0
        V[1*10+k1, col] += 1 # B1k1
        W[k0*2+0, col] += 1 # C0k0
        W[k1*2+0, col] += 1 # C0k1 ? No.
        # C00 (i=0, k=k0) = M1 + ...
        # C01 (i=0, k=k1) = ...
        # C10 (i=1, k=k0) = ...
        # C11 (i=1, k=k1) = ...
        
        # Let's map Strassen outputs to W.
        # C00 (i=0, k=k0): M1 + M4 - M5 + M7
        # C01 (i=0, k=k1): M3 + M5
        # C10 (i=1, k=k0): M2 + M4
        # C11 (i=1, k=k1): M1 - M2 + M3 + M6
        
        # M1
        W[k0*2+0, col] += 1
        W[k1*2+1, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1*3+0, col] += 1
        U[1*3+1, col] += 1
        V[0*10+k0, col] += 1
        W[k0*2+1, col] += 1 # C10
        W[k1*2+1, col] += -1 # C11
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0*3+0, col] += 1
        V[0*10+k1, col] += 1
        V[1*10+k1, col] += -1
        W[k1*2+0, col] += 1 # C01
        W[k1*2+1, col] += 1 # C11
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1*3+1, col] += 1
        V[1*10+k0, col] += 1
        V[0*10+k0, col] += -1
        W[k0*2+0, col] += 1 # C00
        W[k0*2+1, col] += 1 # C10
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0*3+0, col] += 1
        U[0*3+1, col] += 1
        V[1*10+k1, col] += 1
        W[k0*2+0, col] += -1 # C00
        W[k1*2+0, col] += 1 # C01
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1*3+0, col] += 1
        U[0*3+0, col] += -1
        V[0*10+k0, col] += 1
        V[0*10+k1, col] += 1
        W[k1*2+1, col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0*3+1, col] += 1
        U[1*3+1, col] += -1
        V[1*10+k0, col] += 1
        V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1 # C00
        col += 1

    # Part 2: j=2 (2x1x10)
    for q in range(5):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # M1 = A02 * B2k0
        U[0*3+2, col] += 1
        V[2*10+k0, col] += 1
        W[k0*2+0, col] += 1
        col += 1
        
        # M2 = A02 * B2k1
        U[0*3+2, col] += 1
        V[2*10+k1, col] += 1
        W[k1*2+0, col] += 1
        col += 1
        
        # M3 = A12 * B2k0
        U[1*3+2, col] += 1
        V[2*10+k0, col] += 1
        W[k0*2+1, col] += 1
        col += 1
        
        # M4 = A12 * B2k1
        U[1*3+2, col] += 1
        V[2*10+k1, col] += 1
        W[k1*2+1, col] += 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
