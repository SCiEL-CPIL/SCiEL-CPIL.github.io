import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    DEADLINE = 140
    
    rank = 66
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A[i0,k0] + A[i1,k1]) * (B[k0,j0] + B[k1,j1])
        U[i0*m + k0, col] = 1
        U[i1*m + k1, col] = 1
        V[k0*p + j0, col] = 1
        V[k1*p + j1, col] = 1
        W[j0*n + i0, col] = 1
        W[j1*n + i0, col] = 1
        W[j1*n + i1, col] = 1
        W[j0*n + i1, col] = -1
        col += 1
        
        # M2 = (A[i1,k0] + A[i1,k1]) * B[k0,j0]
        U[i1*m + k0, col] = 1
        U[i1*m + k1, col] = 1
        V[k0*p + j0, col] = 1
        W[j0*n + i1, col] = 1
        col += 1
        
        # M3 = A[i0,k0] * (B[k1,j1] - B[k0,j1])
        U[i0*m + k0, col] = 1
        V[k1*p + j1, col] = 1
        V[k0*p + j1, col] = -1
        W[j1*n + i0, col] = 1
        col += 1
        
        # M4 = A[i1,k1] * (B[k1,j0] - B[k0,j0])
        U[i1*m + k1, col] = 1
        V[k1*p + j0, col] = 1
        V[k0*p + j0, col] = -1
        W[j0*n + i0, col] = -1
        W[j0*n + i1, col] = 1
        col += 1
        
        # M5 = (A[i0,k0] + A[i0,k1]) * B[k1,j1]
        U[i0*m + k0, col] = 1
        U[i0*m + k1, col] = 1
        V[k1*p + j1, col] = 1
        W[j1*n + i0, col] = -1
        W[j1*n + i1, col] = 1
        col += 1
        
        # M6 = (A[i1,k0] - A[i0,k0]) * (B[k0,j0] + B[k1,j0])
        U[i1*m + k0, col] = 1
        U[i0*m + k0, col] = -1
        V[k0*p + j0, col] = 1
        V[k1*p + j0, col] = 1
        W[j0*n + i1, col] = 1
        col += 1
        
        # M7 = (A[i0,k1] - A[i1,k1]) * (B[k1,j1] + B[k0,j1])
        U[i0*m + k1, col] = 1
        U[i1*m + k1, col] = -1
        V[k1*p + j1, col] = 1
        V[k0*p + j1, col] = 1
        W[j1*n + i0, col] = 1
        col += 1

    def add_schoolbook_1x2(i, j0, j1, k0, k1):
        nonlocal col
        # k=0, j=j0
        U[i*m + k0, col] = 1
        V[k0*p + j0, col] = 1
        W[j0*n + i, col] = 1
        col += 1
        # k=0, j=j1
        U[i*m + k0, col] = 1
        V[k0*p + j1, col] = 1
        W[j1*n + i, col] = 1
        col += 1
        # k=1, j=j0
        U[i*m + k1, col] = 1
        V[k1*p + j0, col] = 1
        W[j0*n + i, col] = 1
        col += 1
        # k=1, j=j1
        U[i*m + k1, col] = 1
        V[k1*p + j1, col] = 1
        W[j1*n + i, col] = 1
        col += 1

    # Block 1: A[0:2, :] and B[:, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 2: A[0:2, :] and B[:, 2:4]
    add_strassen(0, 1, 2, 3, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Block 3: A[0:2, :] and B[:, 4:6]
    add_strassen(0, 1, 4, 5, 0, 1)
    add_strassen(0, 1, 4, 5, 2, 3)
    
    # Block 4: A[2:3, :] and B[:, 0:2]
    add_schoolbook_1x2(2, 0, 1, 0, 1)
    add_schoolbook_1x2(2, 0, 1, 2, 3)
    
    # Block 5: A[2:3, :] and B[:, 2:4]
    add_schoolbook_1x2(2, 2, 3, 0, 1)
    add_schoolbook_1x2(2, 2, 3, 2, 3)
    
    # Block 6: A[2:3, :] and B[:, 4:6]
    add_schoolbook_1x2(2, 4, 5, 0, 1)
    add_schoolbook_1x2(2, 4, 5, 2, 3)
    
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
