import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for b in range(5):
        # B block b: columns 2b, 2b+1
        b00 = 0 * 10 + 2 * b
        b01 = 0 * 10 + 2 * b + 1
        b10 = 1 * 10 + 2 * b
        b11 = 1 * 10 + 2 * b + 1
        b20 = 2 * 10 + 2 * b
        b21 = 2 * 10 + 2 * b + 1
        
        # C indices for this block:
        # W[k*n+i, r]. C[i,k].
        # k=2b, i=0 -> 4b
        # k=2b, i=1 -> 4b+1
        # k=2b+1, i=0 -> 4b+2
        # k=2b+1, i=1 -> 4b+3
        c_0_2b = 4 * b
        c_1_2b = 4 * b + 1
        c_0_2b1 = 4 * b + 2
        c_1_2b1 = 4 * b + 3
        
        # Strassen for A_L B_L (2x2 x 2)
        # M1: (A00+A11)(B00+B11)
        U[0, col] = 1; U[4, col] = 1
        V[b00, col] = 1; V[b11, col] = 1
        W[c_0_2b, col] = 1; W[c_1_2b1, col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[3, col] = 1; U[4, col] = 1
        V[b00, col] = 1
        W[c_1_2b, col] = 1; W[c_1_2b1, col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[0, col] = 1
        V[b01, col] = 1; V[b11, col] = -1
        W[c_0_2b1, col] = 1; W[c_1_2b1, col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[4, col] = 1
        V[b10, col] = 1; V[b00, col] = -1
        W[c_0_2b, col] = 1; W[c_1_2b, col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[b11, col] = 1
        W[c_0_2b, col] = -1; W[c_0_2b1, col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[3, col] = 1; U[0, col] = -1
        V[b00, col] = 1; V[b01, col] = 1
        W[c_1_2b1, col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, col] = 1; U[4, col] = -1
        V[b10, col] = 1; V[b11, col] = 1
        W[c_0_2b, col] = 1
        col += 1
        
        # Cross terms A_R B_R (2x1 x 2)
        # P1: A02 B20
        U[2, col] = 1; V[b20, col] = 1; W[c_0_2b, col] = 1
        col += 1
        
        # P2: A02 B21
        U[2, col] = 1; V[b21, col] = 1; W[c_0_2b1, col] = 1
        col += 1
        
        # P3: A12 B20
        U[5, col] = 1; V[b20, col] = 1; W[c_1_2b, col] = 1
        col += 1
        
        # P4: A12 B21
        U[5, col] = 1; V[b21, col] = 1; W[c_1_2b1, col] = 1
        col += 1
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
