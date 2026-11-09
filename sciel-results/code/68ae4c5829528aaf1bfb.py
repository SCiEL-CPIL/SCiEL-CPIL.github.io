import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 6
    
    # Strassen 2x2x2
    # U1, V1, W1 shapes (4, 7)
    U1 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 0, 0, 0, 0, -1, 0],# A10 (Wait, Strassen coeffs vary)
        [1, 1, 0, 1, 0, 1, -1] # A11
    ], dtype=float)
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B10
    # M6 = (A11-A10)B01
    # M7 = (A01-A11)B10 -- Wait, M7 is (A01-A11)B10? No.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Let's define U1, V1, W1 based on these.
    # U1 (A coeffs):
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: 0, 0, -1, 1
    # M7: 0, 1, 0, -1
    U1 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 1, -1]
    ], dtype=float)
    
    # V1 (B coeffs):
    # M1: 1, 0, 0, 1
    # M2: 1, 0, 0, 0
    # M3: 0, 1, 0, -1
    # M4: -1, 0, 1, 0
    # M5: 0, 0, 1, 0
    # M6: 0, 1, 0, 0
    # M7: 0, 0, 1, 0
    V1 = np.array([
        [1, 1, 0, -1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 0, -1, 0, 0, 0, 0]
    ], dtype=float)
    
    # W1 (C coeffs):
    # C00: M1(1), M4(1), M5(-1), M7(-1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M6(1), M7(1)
    # W1 index k*n+i. n=2.
    # C00: idx 0*2+0 = 0
    # C01: idx 1*2+0 = 2
    # C10: idx 0*2+1 = 1
    # C11: idx 1*2+1 = 3
    W1 = np.zeros((4, 7))
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = -1
    W1[2, 2] = 1; W1[2, 4] = 1
    W1[1, 1] = 1; W1[1, 3] = 1
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 5] = 1; W1[3, 6] = 1
    
    # Schoolbook 2x2x3
    # U2 (4x12), V2 (6x12), W2 (6x12)
    U2 = np.zeros((4, 12))
    V2 = np.zeros((6, 12))
    W2 = np.zeros((6, 12))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                idx = i*6 + j*3 + k
                U2[i*2+j, idx] = 1
                V2[j*3+k, idx] = 1
                W2[k*2+i, idx] = 1 # n=2, p=3. W[k*n+i]
                
    # Kronecker Product
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Check shapes
    # U: 16x84
    # V: 24x84
    # W: 24x84
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
