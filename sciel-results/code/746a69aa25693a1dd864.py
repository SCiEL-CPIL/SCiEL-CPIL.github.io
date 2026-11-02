import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 83 decomposition using Strassen for 2x2x2 block
    # Schoolbook rank is 84. We replace the 8 terms for i in {0,1}, j in {0,1}, k in {0,1}
    # with 7 Strassen terms.
    
    rank = 83
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    
    # Define the Strassen terms for the 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0*4+0, r] = 1.0; U[1*4+1, r] = 1.0
    V[0*7+0, r] = 1.0; V[1*7+1, r] = 1.0
    W[0*3+0, r] = 1.0; W[1*3+1, r] = 1.0 # C00 and C11
    r += 1
    
    # M2 = (A10 + A11) * B00
    # A10 is i=1, j=0. A11 is i=1, j=1.
    U[1*4+0, r] = 1.0; U[1*4+1, r] = 1.0
    V[0*7+0, r] = 1.0
    W[0*3+1, r] = 1.0 # C10
    r += 1
    
    # M3 = A00 * (B01 - B11)
    U[0*4+0, r] = 1.0
    V[0*7+1, r] = 1.0; V[1*7+1, r] = -1.0
    W[1*3+0, r] = 1.0 # C01
    r += 1
    
    # M4 = A11 * (B10 - B00)
    U[1*4+1, r] = 1.0
    V[1*7+0, r] = 1.0; V[0*7+0, r] = -1.0
    W[0*3+1, r] = 1.0; W[1*3+1, r] = -1.0 # C10 and -C11
    r += 1
    
    # M5 = (A00 + A01) * B11
    U[0*4+0, r] = 1.0; U[0*4+1, r] = 1.0
    V[1*7+1, r] = 1.0
    W[1*3+0, r] = -1.0; W[1*3+1, r] = 1.0 # -C01 and C11
    r += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[1*4+0, r] = 1.0; U[0*4+0, r] = -1.0
    V[0*7+0, r] = 1.0; V[0*7+1, r] = 1.0
    W[0*3+1, r] = 1.0; W[1*3+1, r] = 1.0 # C10 and C11
    r += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[0*4+1, r] = 1.0; U[1*4+1, r] = -1.0
    V[1*7+0, r] = 1.0; V[1*7+1, r] = 1.0
    W[0*3+0, r] = 1.0; W[1*3+0, r] = 1.0 # C00 and C01
    r += 1
    
    # Now add the schoolbook terms for all other (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u, r] = 1.0
                V[idx_v, r] = 1.0
                W[idx_w, r] = 1.0
                r += 1
                
    # r should be 83
    assert r == rank, f"Rank mismatch: {r} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

