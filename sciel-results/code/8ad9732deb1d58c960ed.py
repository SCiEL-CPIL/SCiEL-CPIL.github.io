"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    for block in range(3):
        k0 = block * 2
        
        # Strassen indices:
        # A: a00=0, a01=1, a10=3, a11=4
        # B: b00=k0, b01=k0+1, b10=6+k0, b11=7+k0
        # C: c00=3*k0, c01=3*k0+1, c10=3*(k0+1), c11=3*(k0+1)+1
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to: C00(+1), C11(+1)
        U[0, col] = 1; U[4, col] = 1
        V[k0, col] = 1; V[7+k0, col] = 1
        W[3*k0, col] = 1; W[3*(k0+1)+1, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # Contributes to: C10(+1), C11(-1)
        U[3, col] = 1; U[4, col] = 1
        V[k0, col] = 1
        W[3*k0+1, col] = 1; W[3*(k0+1)+1, col] = -1
        col += 1
        
        # M3 = A00(B10-B00)
        # Contributes to: C01(+1), C11(+1)
        U[0, col] = 1
        V[6+k0, col] = 1; V[k0, col] = -1
        W[3*(k0+1), col] = 1; W[3*(k0+1)+1, col] = 1
        col += 1
        
        # M4 = A11(B01-B10)
        # Contributes to: C00(+1), C10(+1)
        U[4, col] = 1
        V[k0+1, col] = 1; V[6+k0, col] = -1
        W[3*k0, col] = 1; W[3*k0+1, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        # Contributes to: C01(+1), C00(-1)
        U[0, col] = 1; U[1, col] = 1
        V[7+k0, col] = 1
        W[3*(k0+1), col] = 1; W[3*k0, col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        # Contributes to: C11(+1)
        U[3, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[6+k0, col] = 1
        W[3*(k0+1)+1, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        # Contributes to: C00(+1), C10(+1)
        U[1, col] = 1; U[4, col] = -1
        V[k0+1, col] = 1; V[7+k0, col] = 1
        W[3*k0, col] = 1; W[3*k0+1, col] = 1
        col += 1
        
        # Tail: 6 terms for C[k,2]
        for j in range(3):
            for k in range(2):
                kk = k0 + k
                U[6+j, col] = 1
                V[j*6+kk, col] = 1
                W[kk*3+2, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
