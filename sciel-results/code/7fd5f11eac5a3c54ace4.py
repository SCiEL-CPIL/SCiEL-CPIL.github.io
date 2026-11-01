import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    R = 18
    U = [[0] * R for _ in range(4)]
    V = [[0] * R for _ in range(10)]
    W = [[0] * R for _ in range(10)]
    
    def add_strassen(col0, col1, base):
        # P1
        U[0][base] = 1; U[3][base] = 1
        V[0*p+col0][base] = 1
        V[1*p+col1][base] = 1
        W[col0*n+0][base] = 1
        W[col1*n+1][base] = 1
        
        # P2
        U[2][base+1] = 1; U[3][base+1] = 1
        V[0*p+col0][base+1] = 1
        W[col0*n+1][base+1] = 1
        W[col1*n+1][base+1] = -1
        
        # P3
        U[0][base+2] = 1
        V[1*p+col0][base+2] = 1
        V[0*p+col0][base+2] = -1
        W[col1*n+0][base+2] = 1
        W[col1*n+1][base+2] = 1
        
        # P4
        U[3][base+3] = 1
        V[0*p+col1][base+3] = 1
        V[1*p+col1][base+3] = -1
        W[col0*n+0][base+3] = 1
        W[col0*n+1][base+3] = 1
        
        # P5
        U[0][base+4] = 1; U[1][base+4] = 1
        V[1*p+col1][base+4] = 1
        W[col0*n+0][base+4] = -1
        W[col1*n+0][base+4] = 1
        
        # P6
        U[2][base+5] = 1; U[0][base+5] = -1
        V[0*p+col0][base+5] = 1
        V[0*p+col1][base+5] = 1
        W[col1*n+1][base+5] = 1
        
        # P7
        U[1][base+6] = 1; U[3][base+6] = -1
        V[1*p+col0][base+6] = 1
        V[1*p+col1][base+6] = 1
        W[col0*n+0][base+6] = 1

    add_strassen(0, 1, 0)
    add_strassen(2, 3, 7)
    
    # Schoolbook for col 4
    U[0][14] = 1; V[0*p+4][14] = 1; W[4*n+0][14] = 1
    U[1][15] = 1; V[1*p+4][15] = 1; W[4*n+0][15] = 1
    U[2][16] = 1; V[0*p+4][16] = 1; W[4*n+1][16] = 1
    U[3][17] = 1; V[1*p+4][17] = 1; W[4*n+1][17] = 1
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
