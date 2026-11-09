import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a tensor decomposition of matrix multiplication (2x3) * (3x10) -> (2x10).
    Uses block structure with Strassen for 2x2x2 parts and standard for 2x1x2 parts.
    Total rank: 55.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # U: (n*m) x rank = 6 x 55
    # V: (m*p) x rank = 30 x 55
    # W: (p*n) x rank = 20 x 55
    
    total_rank = 55
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2: A[0:2, 0:2] * B[0:2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
        # B indices: (0,c0)->0*p+c0, (0,c1)->0*p+c1, (1,c0)->1*p+c0, (1,c1)->1*p+c1
        # C indices: (0,c0)->c0*n+0, (0,c1)->c1*n+0, (1,c0)->c0*n+1, (1,c1)->c1*n+1
        
        # P1 = (a00 + a11)(b00 + b11)
        U[0][col] = 1
        U[4][col] = 1
        V[0 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P2 = (a10 + a11)b00
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        W[c1 * n + 1][col] = -1
        col += 1
        
        # P3 = a00(b01 - b11)
        U[0][col] = 1
        V[0 * p + c1][col] = 1
        V[1 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P4 = a11(b10 - b00)
        U[4][col] = 1
        V[1 * p + c0][col] = 1
        V[0 * p + c0][col] = -1
        W[c0 * n + 0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # P5 = (a00 + a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = -1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # P6 = (a10 - a00)(b00 + b01)
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + c0][col] = 1
        V[0 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # P7 = (a01 - a11)(b10 + b11)
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + c0][col] = 1
        V[1 * p + c1][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
        # Standard 2x1x2: A[0:2, 2] * B[2, c0:c1+1] -> C[0:2, c0:c1+1]
        # A[0,2]=U[2], A[1,2]=U[5]
        # B[2,c0]=V[2*p+c0], B[2,c1]=V[2*p+c1]
        
        # Term 1: A[0,2] * B[2,c0] -> C[0,c0]
        U[2][col] = 1
        V[2 * p + c0][col] = 1
        W[c0 * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,2] * B[2,c1] -> C[0,c1]
        U[2][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,2] * B[2,c0] -> C[1,c0]
        U[5][col] = 1
        V[2 * p + c0][col] = 1
        W[c0 * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,2] * B[2,c1] -> C[1,c1]
        U[5][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
