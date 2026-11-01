import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 40
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # A indices
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        b00 = 0 * 10 + k0
        b01 = 0 * 10 + k1
        b10 = 1 * 10 + k0
        b11 = 1 * 10 + k1
        b20 = 2 * 10 + k0
        b21 = 2 * 10 + k1
        
        # C indices
        c00 = k0 * 2 + 0
        c01 = k1 * 2 + 0
        c10 = k0 * 2 + 1
        c11 = k1 * 2 + 1
        
        # Strassen terms for j=0,1
        # M1
        U[a00, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M2
        U[a10, col] = 1.0
        U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = -1.0
        col += 1
        
        # M3
        U[a00, col] = 1.0
        V[b01, col] = 1.0
        V[b11, col] = -1.0
        W[c01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M4
        U[a11, col] = 1.0
        V[b10, col] = 1.0
        V[b00, col] = -1.0
        W[c00, col] = 1.0
        W[c10, col] = 1.0
        col += 1
        
        # M5
        U[a00, col] = 1.0
        U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0
        W[c01, col] = 1.0
        col += 1
        
        # M6
        U[a10, col] = 1.0
        U[a00, col] = -1.0
        V[b00, col] = 1.0
        V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
        # M7
        U[a01, col] = 1.0
        U[a11, col] = -1.0
        V[b10, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1
        
        # Extra term for j=2
        U[a02, col] = 1.0
        U[a12, col] = 1.0
        V[b20, col] = 1.0
        V[b21, col] = 1.0
        W[c00, col] = 1.0
        W[c01, col] = 1.0
        W[c10, col] = 1.0
        W[c11, col] = 1.0
        col += 1
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
