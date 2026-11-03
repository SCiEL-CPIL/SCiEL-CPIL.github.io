import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 189 Decomposition: Schoolbook 3x3 (27) x Strassen 2x2 (7)
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Strassen 2x2 Coefficients
    # U2[a,b,s], V2[b,c,s], W2[a,c,s]
    # s in 0..6
    U2 = np.zeros((2, 2, 7), dtype=int)
    V2 = np.zeros((2, 2, 7), dtype=int)
    W2 = np.zeros((2, 2, 7), dtype=int)
    
    # M1 = A00 (B01 - B11)
    U2[0, 0, 0] = 1
    V2[0, 1, 0] = 1
    V2[1, 1, 0] = -1
    W2[0, 0, 0] = 1
    W2[1, 0, 0] = 1
    W2[1, 1, 0] = 1
    
    # M2 = (A00 + A10) B11
    U2[0, 0, 1] = 1
    U2[1, 0, 1] = 1
    V2[1, 1, 1] = 1
    W2[0, 0, 1] = -1
    W2[1, 0, 1] = 1
    
    # M3 = (A10 - A00) B00
    U2[1, 0, 2] = 1
    U2[0, 0, 2] = -1
    V2[0, 0, 2] = 1
    W2[0, 0, 2] = 1
    W2[1, 0, 2] = 1
    
    # M4 = A11 (B10 - B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1
    V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1
    W2[1, 1, 3] = 1
    
    # M5 = (A00 + A11) (B00 + B11)
    U2[0, 0, 4] = 1
    U2[1, 1, 4] = 1
    V2[0, 0, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = 1
    W2[0, 1, 4] = 1
    W2[1, 0, 4] = 1
    W2[1, 1, 4] = 1
    
    # M6 = (A01 - A11) (B10 + B11)
    U2[0, 1, 5] = 1
    U2[1, 1, 5] = -1
    V2[1, 0, 5] = 1
    V2[1, 1, 5] = 1
    W2[0, 0, 5] = 1
    W2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # M7 = (A00 + A01) (B00 - B10)
    U2[0, 0, 6] = 1
    U2[0, 1, 6] = 1
    V2[0, 0, 6] = 1
    V2[1, 0, 6] = -1
    W2[0, 1, 6] = 1
    W2[1, 0, 6] = 1
    W2[1, 1, 6] = 1
    
    col = 0
    # Schoolbook 3x3: Iterate over block indices I, K, J
    for I in range(3):
        for K in range(3):
            for J in range(3):
                # For each Strassen term s
                for s in range(7):
                    # U: A[I, K] block
                    # A element (i, j) where i = I*2 + a, j = K*2 + b
                    for a in range(2):
                        for b in range(2):
                            u_val = U2[a, b, s]
                            if u_val != 0:
                                i = I * 2 + a
                                j = K * 2 + b
                                U[i * 6 + j, col] += u_val
                                
                    # V: B[K, J] block
                    # B element (k, l) where k = K*2 + b, l = J*2 + c
                    for b in range(2):
                        for c in range(2):
                            v_val = V2[b, c, s]
                            if v_val != 0:
                                k = K * 2 + b
                                l = J * 2 + c
                                V[k * 6 + l, col] += v_val
                                
                    # W: C[I, J] block
                    # C element (i, l) where i = I*2 + a, l = J*2 + c
                    # Index convention: W[row * 6 + col]
                    for a in range(2):
                        for c in range(2):
                            w_val = W2[a, c, s]
                            if w_val != 0:
                                i = I * 2 + a
                                l = J * 2 + c
                                W[i * 6 + l, col] += w_val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
