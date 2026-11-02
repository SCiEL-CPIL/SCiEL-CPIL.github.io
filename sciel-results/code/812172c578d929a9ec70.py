import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # g corresponds to middle dimension split (2 blocks of size 2)
    # h corresponds to output dimension split (4 blocks of size 2)
    for g in range(2):
        j0 = 2 * g
        j1 = 2 * g + 1
        for h in range(4):
            k0 = 2 * h
            k1 = 2 * h + 1
            
            # Strassen Term 1: (X00 + X11)(Y00 + Y11)
            # U: (0, j0): 1, (1, j1): 1
            U[2*0 + j0, col] = 1
            U[2*1 + j1, col] = 1
            # V: (j0, k0): 1, (j1, k1): 1
            V[8*j0 + k0, col] = 1
            V[8*j1 + k1, col] = 1
            # W: Z00 += M1, Z11 += M1
            # W row for (k0, 0) is 2*k0 + 0
            # W row for (k1, 1) is 2*k1 + 1
            W[2*k0 + 0, col] = 1
            W[2*k1 + 1, col] = 1
            col += 1
            
            # Strassen Term 2: (X10 + X11)Y00
            # U: (1, j0): 1, (1, j1): 1
            U[2*1 + j0, col] = 1
            U[2*1 + j1, col] = 1
            # V: (j0, k0): 1
            V[8*j0 + k0, col] = 1
            # W: Z10 += M2, Z11 -= M2
            # W row for (k0, 1) is 2*k0 + 1
            # W row for (k1, 1) is 2*k1 + 1
            W[2*k0 + 1, col] = 1
            W[2*k1 + 1, col] = -1
            col += 1
            
            # Strassen Term 3: X00(Y01 - Y11)
            # U: (0, j0): 1
            U[2*0 + j0, col] = 1
            # V: (j0, k1): 1, (j1, k1): -1
            V[8*j0 + k1, col] = 1
            V[8*j1 + k1, col] = -1
            # W: Z01 += M3, Z11 += M3
            # W row for (k1, 0) is 2*k1 + 0
            # W row for (k1, 1) is 2*k1 + 1
            W[2*k1 + 0, col] = 1
            W[2*k1 + 1, col] = 1
            col += 1
            
            # Strassen Term 4: X11(Y10 - Y00)
            # U: (1, j1): 1
            U[2*1 + j1, col] = 1
            # V: (j1, k0): 1, (j0, k0): -1
            V[8*j1 + k0, col] = 1
            V[8*j0 + k0, col] = -1
            # W: Z00 += M4, Z10 += M4
            # W row for (k0, 0) is 2*k0 + 0
            # W row for (k0, 1) is 2*k0 + 1
            W[2*k0 + 0, col] = 1
            W[2*k0 + 1, col] = 1
            col += 1
            
            # Strassen Term 5: (X00 + X01)Y11
            # U: (0, j0): 1, (0, j1): 1
            U[2*0 + j0, col] = 1
            U[2*0 + j1, col] = 1
            # V: (j1, k1): 1
            V[8*j1 + k1, col] = 1
            # W: Z00 -= M5, Z01 += M5
            # W row for (k0, 0) is 2*k0 + 0
            # W row for (k1, 0) is 2*k1 + 0
            W[2*k0 + 0, col] = -1
            W[2*k1 + 0, col] = 1
            col += 1
            
            # Strassen Term 6: (X10 - X00)(Y00 + Y01)
            # U: (1, j0): 1, (0, j0): -1
            U[2*1 + j0, col] = 1
            U[2*0 + j0, col] = -1
            # V: (j0, k0): 1, (j0, k1): 1
            V[8*j0 + k0, col] = 1
            V[8*j0 + k1, col] = 1
            # W: Z11 += M6
            # W row for (k1, 1) is 2*k1 + 1
            W[2*k1 + 1, col] = 1
            col += 1
            
            # Strassen Term 7: (X01 - X11)(Y10 + Y11)
            # U: (0, j1): 1, (1, j1): -1
            U[2*0 + j1, col] = 1
            U[2*1 + j1, col] = -1
            # V: (j1, k0): 1, (j1, k1): 1
            V[8*j1 + k0, col] = 1
            V[8*j1 + k1, col] = 1
            # W: Z00 += M7
            # W row for (k0, 0) is 2*k0 + 0
            W[2*k0 + 0, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
