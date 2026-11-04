import time
import numpy as np

def _strassen_2x2():
    # Returns U, V, W for 2x2x2 matmul with rank 7
    # Strassen's algorithm
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # U: A11=1, A22=1
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=1, B22=1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11=1, C22=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11 -> C21 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21 is index 2 (k=1, i=1 -> 1*2+1=3? No. W[k*2+i]. C21 is i=1, k=1? No. C[i,k]. i=1, k=1? 
    # Let's check indices.
    # C is 2x2. C00, C01, C10, C11.
    # W index is k*2+i.
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 2.
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # M1: C00 and C11.
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: C10.
    W[1, 1] = 1.
    
    # M3: A11(B12-B22) -> C01 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01 is index 2.
    
    # M4: A22(B21-B11) -> C11 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1 # C11 is index 3.
    
    # M5: (A11+A12)B22 -> C01 += M5, C11 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1; W[3, 4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C10 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C01 += M7, C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Fallback to schoolbook for safety if time runs out or verification fails
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
