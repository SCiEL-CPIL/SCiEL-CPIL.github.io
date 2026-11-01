import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # Using Strassen-style decomposition for 2x2 blocks
    # Each 2x2 block of B (columns 2k, 2k+1) is multiplied with A using 7 products
    # Total rank = 7 blocks * 7 products = 49
    rank = 7 * (p // 2)  # p=14, so 7 blocks
    
    # U: shape (n*m, rank) = (4, 49)
    # V: shape (m*p, rank) = (28, 49)
    # W: shape (p*n, rank) = (28, 49)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen forms for A:
    # L1 = a + d       -> indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # L2 = c + d
    # L3 = a
    # L4 = d
    # L5 = a + b
    # L6 = c - a
    # L7 = b - d
    
    # Strassen forms for B (each 2x2 block):
    # K1 = e + h       -> B[0,2k]=e, B[0,2k+1]=f, B[1,2k]=g, B[1,2k+1]=h
    # K2 = e
    # K3 = g - h
    # K4 = f - e
    # K5 = h
    # K6 = e + f
    # K7 = g + h
    
    # Reconstruction:
    # C[0,2k]   = M1 + M4 - M5 + M7
    # C[0,2k+1] = M3 + M5
    # C[1,2k]   = M2 + M4
    # C[1,2k+1] = M1 - M2 + M3 + M6
    
    # A forms (U columns): U[i*m + j, c] = coefficient of A[i,j] in L_r
    # For each product r (0-indexed):
    # r=0: L1 = a + d -> U[0,0]=1, U[3,0]=1
    # r=1: L2 = c + d -> U[2,1]=1, U[3,1]=1
    # r=2: L3 = a     -> U[0,2]=1
    # r=3: L4 = d     -> U[3,3]=1
    # r=4: L5 = a + b -> U[0,4]=1, U[1,4]=1
    # r=5: L6 = c - a -> U[2,5]=1, U[0,5]=-1
    # r=6: L7 = b - d -> U[1,6]=1, U[3,6]=-1
    
    a_forms = [
        [1, 0, 1, 0, 1, -1, 0],   # a (A[0,0])
        [0, 0, 0, 0, 1, 0, 1],    # b (A[0,1])
        [0, 1, 0, 0, 0, 1, 0],    # c (A[1,0])
        [1, 1, 0, 1, 0, 0, -1]    # d (A[1,1])
    ]
    
    # B forms for each block k:
    # K1 = e + h       -> V[0*14+2k]=1, V[1*14+2k+1]=1  (i.e., row 2k and row 14+2k+1)
    # K2 = e           -> V[2k]=1
    # K3 = g - h       -> V[14+2k]=1, V[14+2k+1]=-1
    # K4 = f - e       -> V[2k+1]=1, V[2k]=-1
    # K5 = h           -> V[14+2k+1]=1
    # K6 = e + f       -> V[2k]=1, V[2k+1]=1
    # K7 = g + h       -> V[14+2k]=1, V[14+2k+1]=1
    
    for k in range(p // 2):
        for r in range(7):
            c = k * 7 + r
            
            # Fill U
            for j in range(4):  # j is i*m + j where i in {0,1}, j in {0,1}
                U[j, c] = a_forms[j][r]
            
            # Fill V
            col_e = 2 * k       # B[0, 2k]
            col_f = 2 * k + 1   # B[0, 2k+1]
            col_g = 14 + 2 * k       # B[1, 2k]
            col_h = 14 + 2 * k + 1   # B[1, 2k+1]
            
            if r == 0:    # K1 = e + h
                V[col_e, c] = 1
                V[col_h, c] = 1
            elif r == 1:  # K2 = e
                V[col_e, c] = 1
            elif r == 2:  # K3 = g - h
                V[col_g, c] = 1
                V[col_h, c] = -1
            elif r == 3:  # K4 = f - e
                V[col_f, c] = 1
                V[col_e, c] = -1
            elif r == 4:  # K5 = h
                V[col_h, c] = 1
            elif r == 5:  # K6 = e + f
                V[col_e, c] = 1
                V[col_f, c] = 1
            elif r == 6:  # K7 = g + h
                V[col_g, c] = 1
                V[col_h, c] = 1
    
    # Fill W: W[k'*n + i, c] = coefficient of product r in C[i, k']
    # C[0, 2k]   = M1 + M4 - M5 + M7  -> W[0, c] for r in {0,3,4,6}: W[0,c] += [1,0,0,1,-1,0,1][r]
    # C[0, 2k+1] = M3 + M5            -> W[1, c] for r in {2,4}: W[1,c] += [0,0,1,0,1,0,0][r]
    # C[1, 2k]   = M2 + M4            -> W[14+0, c] for r in {1,3}: W[14,c] += [0,1,0,1,0,0,0][r]
    # C[1, 2k+1] = M1 - M2 + M3 + M6  -> W[14+1, c] for r in {0,1,2,5}: W[15,c] += [1,-1,1,0,0,1,0][r]
    
    for k in range(p // 2):
        for r in range(7):
            c = k * 7 + r
            # W index: k'*n + i where k' is column of B (0..13), i is row of A (0..1)
            # C[i, k'] -> W[k'*n + i, c]
            
            # C[0, 2k] -> W[2k*2 + 0, c] = W[4k, c]
            if r == 0: W[4*k, c] += 1
            elif r == 3: W[4*k, c] += 1
            elif r == 4: W[4*k, c] += -1
            elif r == 6: W[4*k, c] += 1
            
            # C[0, 2k+1] -> W[(2k+1)*2 + 0, c] = W[4k+2, c]
            if r == 2: W[4*k+2, c] += 1
            elif r == 4: W[4*k+2, c] += 1
            
            # C[1, 2k] -> W[2k*2 + 1, c] = W[4k+1, c]
            if r == 1: W[4*k+1, c] += 1
            elif r == 3: W[4*k+1, c] += 1
            
            # C[1, 2k+1] -> W[(2k+1)*2 + 1, c] = W[4k+3, c]
            if r == 0: W[4*k+3, c] += 1
            elif r == 1: W[4*k+3, c] += -1
            elif r == 2: W[4*k+3, c] += 1
            elif r == 5: W[4*k+3, c] += 1
    
    # Convert to lists as expected
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
