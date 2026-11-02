"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: 7 Strassen blocks (rank 7) for columns 0-13, 
    # and 1 Schoolbook block (rank 4) for column 14.
    # Total Rank = 53.
    
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_strassen(k0, k1, offset):
        # Local indices for A: 0,1,2,3
        # Local indices for B: 0..14, 15..29
        # Local indices for C: 0..29
        
        r = offset
        # P1 = (a+d)(e+h)
        U[0, r] += 1.0; U[3, r] += 1.0
        V[0, k0] += 1.0; V[15, k1] += 1.0
        W[2*k0, r] += 1.0; W[2*k1+1, r] += 1.0
        
        # P2 = (c+d)e
        U[2, r+1] += 1.0; U[3, r+1] += 1.0
        V[0, k0] += 1.0
        W[2*k0+1, r+1] += 1.0; W[2*k1+1, r+1] -= 1.0
        
        # P3 = a(f-h)
        U[0, r+2] += 1.0
        V[0, k1] += 1.0; V[15, k1] -= 1.0
        W[2*k1, r+2] += 1.0; W[2*k1+1, r+2] += 1.0
        
        # P4 = d(g-e)
        U[3, r+3] += 1.0
        V[15, k0] += 1.0; V[0, k0] -= 1.0
        W[2*k0, r+3] += 1.0; W[2*k0+1, r+3] += 1.0
        
        # P5 = (a+b)h
        U[0, r+4] += 1.0; U[1, r+4] += 1.0
        V[15, k1] += 1.0
        W[2*k0, r+4] -= 1.0; W[2*k1, r+4] += 1.0
        
        # P6 = (c-a)(e+f)
        U[2, r+5] += 1.0; U[0, r+5] -= 1.0
        V[0, k0] += 1.0; V[0, k1] += 1.0
        W[2*k1+1, r+5] += 1.0
        
        # P7 = (b-d)(g+h)
        U[1, r+6] += 1.0; U[3, r+6] -= 1.0
        V[15, k0] += 1.0; V[15, k1] += 1.0
        W[2*k0, r+6] += 1.0

    # Apply 7 Strassen blocks
    for i in range(7):
        set_strassen(2*i, 2*i+1, 7*i)
        
    # Apply Schoolbook for last column (14)
    base = 49
    U[0, base] = 1.0
    V[0, 14] = 1.0
    W[28, base] = 1.0 # C[0, 14] -> index 14*2 + 0 = 28
    
    U[1, base+1] = 1.0
    V[15, 14] = 1.0
    W[28, base+1] = 1.0
    
    U[2, base+2] = 1.0
    V[0, 14] = 1.0
    W[29, base+2] = 1.0 # C[1, 14] -> index 14*2 + 1 = 29
    
    U[3, base+3] = 1.0
    V[15, 14] = 1.0
    W[29, base+3] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
