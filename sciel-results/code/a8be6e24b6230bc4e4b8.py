import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    # Split p=5 into 2, 2, 1
    # Strassen for 2x2x2 (rank 7) twice
    # Schoolbook for 2x2x1 (rank 4) once
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # A is 2x2, B is 2x2 (cols c0, c1)
    # Outputs C00, C01, C10, C11
    def add_strassen(offset, c0, c1):
        # S1 = (a00+a11)(b00+b11)
        # S2 = (a10+a11)b01
        # S3 = a00(b10-b11)
        # S4 = a11(b00-b10)
        # S5 = (a00+a01)b11
        # S6 = (a10-a00)(b00+b10)
        # S7 = (a01-a11)(b01+b11)
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # Col 0, 1 of B are c0, c1
        # B00 is V[0*p + c0]
        # B01 is V[0*p + c1]
        # B10 is V[1*p + c0]
        # B11 is V[1*p + c1]
        
        # 1
        U[0*m+0][offset] = 1
        U[1*m+1][offset] = 1
        V[0*p+c0][offset] = 1
        V[1*p+c1][offset] = 1
        W[c0*n+0][offset] = 1
        W[c1*n+0][offset] = 1
        W[c0*n+1][offset] = 1
        W[c1*n+1][offset] = 1
        
        # 2
        U[1*m+0][offset+1] = 1
        U[1*m+1][offset+1] = 1
        V[0*p+c1][offset+1] = 1
        W[c1*n+0][offset+1] = 1
        W[c1*n+1][offset+1] = 1
        
        # 3
        U[0*m+0][offset+2] = 1
        V[1*p+c0][offset+2] = 1
        V[1*p+c1][offset+2] = -1
        W[c0*n+1][offset+2] = 1
        W[c1*n+1][offset+2] = -1
        
        # 4
        U[1*m+1][offset+3] = 1
        V[0*p+c0][offset+3] = 1
        V[1*p+c0][offset+3] = -1
        W[c0*n+0][offset+3] = 1
        W[c0*n+1][offset+3] = -1
        
        # 5
        U[0*m+0][offset+4] = 1
        U[0*m+1][offset+4] = 1
        V[1*p+c1][offset+4] = 1
        W[c1*n+0][offset+4] = 1
        W[c1*n+1][offset+4] = 1
        
        # 6
        U[1*m+0][offset+5] = 1
        U[0*m+0][offset+5] = -1
        V[0*p+c0][offset+5] = 1
        V[1*p+c0][offset+5] = 1
        W[c0*n+0][offset+5] = 1
        W[c0*n+1][offset+5] = 1
        
        # 7
        U[0*m+1][offset+6] = 1
        U[1*m+1][offset+6] = -1
        V[0*p+c1][offset+6] = 1
        V[1*p+c1][offset+6] = 1
        W[c1*n+0][offset+6] = 1
        W[c1*n+1][offset+6] = 1
        
    # Helper to add schoolbook 2x2x1
    def add_schoolbook_1(offset, c):
        # A00 B0c
        U[0*m+0][offset] = 1
        V[0*p+c][offset] = 1
        W[c*n+0][offset] = 1
        
        # A01 B1c
        U[0*m+1][offset+1] = 1
        V[1*p+c][offset+1] = 1
        W[c*n+0][offset+1] = 1
        
        # A10 B0c
        U[1*m+0][offset+2] = 1
        V[0*p+c][offset+2] = 1
        W[c*n+1][offset+2] = 1
        
        # A11 B1c
        U[1*m+1][offset+3] = 1
        V[1*p+c][offset+3] = 1
        W[c*n+1][offset+3] = 1

    # Block 1: cols 0, 1
    add_strassen(0, 0, 1)
    # Block 2: cols 2, 3
    add_strassen(7, 2, 3)
    # Block 3: col 4
    add_schoolbook_1(14, 4)
    
    return {"U": U, "V": V, "W": W}
