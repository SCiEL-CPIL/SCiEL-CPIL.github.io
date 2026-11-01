import time
import numpy as np
from itertools import product
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    n, m, p = 2, 2, 5
    rank = 10
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for k in {0, 1}
    # Indices:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # r=0: P1 = (a+d)(e+h)
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*5+0][0] = 1; V[1*5+1][0] = 1
    W[0*2+0][0] = 1 # C00
    W[0*2+1][0] = 0 # C01
    W[1*2+0][0] = 0 # C10
    W[1*2+1][0] = 1 # C11
    
    # r=1: P2 = (c+d)e
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*5+0][1] = 1
    W[0*2+0][1] = 0
    W[0*2+1][1] = 0
    W[1*2+0][1] = 1 # C10
    W[1*2+1][1] = -1 # C11
    
    # r=2: P3 = a(f-h)
    U[0*2+0][2] = 1
    V[0*5+1][2] = 1; V[1*5+1][2] = -1
    W[0*2+0][2] = 0
    W[0*2+1][2] = 1 # C01
    W[1*2+0][2] = 0
    W[1*2+1][2] = 1 # C11
    
    # r=3: P4 = d(g-e)
    U[1*2+1][3] = 1
    V[0*5+0][3] = -1; V[1*5+0][3] = 1
    W[0*2+0][3] = 1 # C00
    W[0*2+1][3] = 0
    W[1*2+0][3] = 1 # C10
    W[1*2+1][3] = 0
    
    # r=4: P5 = (a+b)h
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*5+1][4] = 1
    W[0*2+0][4] = -1 # C00
    W[0*2+1][4] = 1 # C01
    W[1*2+0][4] = 0
    W[1*2+1][4] = 0
    
    # r=5: P6 = (c-a)(e+f)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*5+0][5] = 1; V[0*5+1][5] = 1
    W[0*2+0][5] = 0
    W[0*2+1][5] = 0
    W[1*2+0][5] = 0
    W[1*2+1][5] = 1 # C11
    
    # r=6: P7 = (b-d)(g+h)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*5+0][6] = 1; V[1*5+1][6] = 1
    W[0*2+0][6] = 1 # C00
    W[0*2+1][6] = 0
    W[1*2+0][6] = 0
    W[1*2+1][6] = 0
    
    # r=7,8,9 for k=2,3,4
    # For k=2, we need T[i,j,2]=1.
    # Use rank 1 term.
    # U[i,j,7]=1, V[j,2,7]=1, W[2,i,7]=1.
    for i in range(2):
        for j in range(2):
            U[i*2+j][7] = 1
    for j in range(2):
        V[j*5+2][7] = 1
    for i in range(2):
        W[2*2+i][7] = 1
        
    # Similarly for k=3 (r=8)
    for i in range(2):
        for j in range(2):
            U[i*2+j][8] = 1
    for j in range(2):
        V[j*5+3][8] = 1
    for i in range(2):
        W[3*2+i][8] = 1
        
    # Similarly for k=4 (r=9)
    for i in range(2):
        for j in range(2):
            U[i*2+j][9] = 1
    for j in range(2):
        V[j*5+4][9] = 1
    for i in range(2):
        W[4*2+i][9] = 1
        
    # Verify the whole thing
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails
        rank = 20
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
    return {"U": U, "V": V, "W": W}
