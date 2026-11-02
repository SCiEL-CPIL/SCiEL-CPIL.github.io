import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 51 decomposition
    # 7 Strassen blocks for columns 0-13 (pairs)
    # 1 Rank 2 block for column 14
    
    rank = 51
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen block for columns k0, k1
    def add_strassen(k0, k1, offset):
        # M1
        U[0, offset] = 1; U[3, offset] = 1
        V[k0, offset] = 1; V[15 + k1, offset] = 1
        W[k0 * 2, offset] = 1; W[k1 * 2 + 1, offset] = 1
        
        # M2
        U[2, offset+1] = 1; U[3, offset+1] = 1
        V[k0, offset+1] = 1
        W[k0 * 2 + 1, offset+1] = 1
        
        # M3
        U[0, offset+2] = 1
        V[15 + k1, offset+2] = 1; V[15 + k1, offset+2] = 1 # Wait, B01 - B11
        # V index for j=0, k=k1 is k1. V index for j=1, k=k1 is 15+k1.
        V[k1, offset+2] = 1
        V[15 + k1, offset+2] = -1
        W[k1 * 2, offset+2] = 1
        
        # M4
        U[3, offset+3] = 1
        V[15 + k0, offset+3] = 1; V[k0, offset+3] = -1
        W[k0 * 2, offset+3] = 1; W[k0 * 2 + 1, offset+3] = 1
        
        # M5
        U[0, offset+4] = 1; U[1, offset+4] = 1
        V[15 + k1, offset+4] = 1
        W[k1 * 2, offset+4] = 1; W[k0 * 2, offset+4] = -1
        
        # M6
        U[2, offset+5] = 1; U[0, offset+5] = -1
        V[k0, offset+5] = 1; V[k1, offset+5] = 1
        W[k1 * 2 + 1, offset+5] = 1
        
        # M7
        U[1, offset+6] = 1; U[3, offset+6] = -1
        V[15 + k0, offset+6] = 1; V[15 + k1, offset+6] = 1
        W[k0 * 2, offset+6] = 1

    # 7 Strassen blocks
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        add_strassen(k0, k1, i * 7)
    
    # Rank 2 block for column 14
    k2 = 14
    offset = 49
    
    # r1: A00 B0k
    U[0, offset] = 1
    U[2, offset] = 1
    V[k2, offset] = 1
    W[k2 * 2, offset] = 1
    W[k2 * 2 + 1, offset] = 1
    
    # r2: A01 B1k
    U[1, offset+1] = 1
    U[3, offset+1] = 1
    V[15 + k2, offset+1] = 1
    W[k2 * 2, offset+1] = 1
    W[k2 * 2 + 1, offset+1] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
