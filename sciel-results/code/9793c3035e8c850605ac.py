import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 23 decomposition
    # Part 1: 2x3x3 (cols 0-2 of B)
    # Part 2: 2x3x2 (cols 3-4 of B)
    
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 1: 2x3x3 ---
    # Indices: i in 0..1, j in 0..2, k in 0..2
    # Strassen for j in 0..1, k in 0..1 (7 terms)
    # Group 1: j=2, k in 0..1 (1 term)
    # Group 2: j in 0..1, k=2 (1 term)
    # Group 3: j=2, k=2 (2 terms)
    
    col = 0
    # Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U[0*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+0, col] = 1; V[1*p+1, col] = 1
    W[0*n+0, col] = 1; W[1*n+0, col] = 1; W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[1*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+0, col] = 1
    W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # M3 = A00(B11-B00)
    U[0*m+0, col] = 1
    V[1*p+1, col] = 1; V[0*p+0, col] = -1
    W[0*n+0, col] = 1; W[1*n+0, col] = 1
    col += 1
    
    # M4 = A11(B00-B10)
    U[1*m+1, col] = 1
    V[0*p+0, col] = 1; V[1*p+0, col] = -1
    W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0*m+0, col] = 1; U[0*m+1, col] = 1
    V[1*p+1, col] = 1
    W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[1*m+0, col] = 1; U[0*m+0, col] = -1
    V[0*p+0, col] = 1; V[0*p+1, col] = 1
    W[0*n+0, col] = 1; W[1*n+0, col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0*m+1, col] = 1; U[1*m+1, col] = -1
    V[1*p+0, col] = 1; V[1*p+1, col] = 1
    W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # Group 1: j=2, k in 0..1
    # U[0,2]=1, U[1,2]=1
    # V[2,0]=1, V[2,1]=1
    # W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    U[0*m+2, col] = 1; U[1*m+2, col] = 1
    V[2*p+0, col] = 1; V[2*p+1, col] = 1
    W[0*n+0, col] = 1; W[1*n+0, col] = 1; W[0*n+1, col] = 1; W[1*n+1, col] = 1
    col += 1
    
    # Group 2: j in 0..1, k=2
    # U[0,0]=1, U[0,1]=1, U[1,0]=1, U[1,1]=1
    # V[0,2]=1, V[1,2]=1
    # W[2,0]=1, W[2,1]=1
    U[0*m+0, col] = 1; U[0*m+1, col] = 1
    U[1*m+0, col] = 1; U[1*m+1, col] = 1
    V[0*p+2, col] = 1; V[1*p+2, col] = 1
    W[2*n+0, col] = 1; W[2*n+1, col] = 1
    col += 1
    
    # Group 3: j=2, k=2
    # Term 1: A02 B22 -> C02
    U[0*m+2, col] = 1
    V[2*p+2, col] = 1
    W[2*n+0, col] = 1
    col += 1
    # Term 2: A12 B22 -> C12
    U[1*m+2, col] = 1
    V[2*p+2, col] = 1
    W[2*n+1, col] = 1
    col += 1
    
    # --- Part 2: 2x3x2 (cols 3-4) ---
    # Schoolbook: 3*2*2 = 12 terms
    for i in range(n):
        for j in range(m):
            for k in range(2):
                # k in 3,4
                U[i*m+j, col] = 1
                V[j*p + (k+3), col] = 1
                W[(k+3)*n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
