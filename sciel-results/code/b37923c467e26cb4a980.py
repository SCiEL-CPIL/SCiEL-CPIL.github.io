import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(3):
        # Block l corresponds to k in {2l, 2l+1}
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # Strassen part: 2x2x2 using A[0:2, 0:2] and B[0:2, k0:k1+1]
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {k0, k1}
        
        # M1 = A22 * (B11 + B22)
        # A22 is i=1, j=1. Index in U: 1*m + 1 = 4
        # B11 is j=0, k=k0. Index in V: 0*p + k0
        # B22 is j=1, k=k1. Index in V: 1*p + k1
        # C11 is i=0, k=k0. Index in W: k0*n + 0
        # C22 is i=1, k=k1. Index in W: k1*n + 1
        
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A11 + A22) * (B11 + B22)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1 # C11 = ... - M2 ...? 
        # Let's re-verify Strassen output formulas.
        
        # Standard Strassen:
        # M1 = A22(B11+B22)
        # M2 = (A11+A22)(B11+B22)
        # M3 = (A11-A22)(B21-B22)
        # M4 = A21(B11)
        # M5 = A11(B22)
        # M6 = (A12-A22)(B21+B22)
        # M7 = (A21-A11)(B11+B12)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 - M1 - M4 + M6
        # C22 = M2 - M3 + M1 - M6
        
        # So:
        # M1: C11 += 1, C22 += 1
        # M2: C21 += 1, C22 += 1
        # M3: C12 += 1, C22 -= 1
        # M4: C11 += 1, C21 -= 1
        # M5: C11 -= 1, C12 += 1
        # M6: C21 += 1, C22 -= 1
        # M7: C11 += 1
        
        # Correcting M2:
        # C21 += 1, C22 += 1
        W[k0 * n + 1][col] = 1 # C21
        W[k1 * n + 1][col] = 1 # C22
        # C11? No.
        col += 1
        
        # M3 = (A11 - A22) * (B21 - B22)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1 # B21 is j=1, k=k0
        V[1 * p + k1][col] = -1 # B22 is j=1, k=k1
        W[k1 * n + 0][col] = 1 # C12
        W[k1 * n + 1][col] = -1 # C22
        col += 1
        
        # M4 = A21 * B11
        U[1 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1 # C11
        W[k0 * n + 1][col] = -1 # C21
        col += 1
        
        # M5 = A11 * B22
        U[0 * m + 0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1 # C11
        W[k1 * n + 0][col] = 1 # C12
        col += 1
        
        # M6 = (A12 - A22) * (B21 + B22)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 1][col] = 1 # C21
        W[k1 * n + 1][col] = -1 # C22
        col += 1
        
        # M7 = (A21 - A11) * (B11 + B12)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1 # C11
        col += 1
        
        # Schoolbook part: 4 terms
        # t8 = A13 * B31
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1 # C11
        col += 1
        
        # t9 = A13 * B32
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1 # C12
        col += 1
        
        # t10 = A23 * B31
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1 # C21
        col += 1
        
        # t11 = A23 * B32
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1 # C22
        col += 1
        
    return {"U": U, "V": V, "W": W}
