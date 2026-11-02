import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _strassen_222():
    # Returns U, V, W for 2x2x2
    # i, j, k in {0, 1}
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Index: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0; A22: i=1,j=1
    # B11: j=0,k=0; B22: j=1,k=1
    # C11: i=0,k=0; C22: i=1,k=1
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to fill U, V, W such that sum U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
    # T[i,j,k] = 1 for all i,j,k.
    
    # r=0: M1
    # A terms: A11 (0,0), A22 (1,1)
    # B terms: B11 (0,0), B22 (1,1)
    # C terms: C11 (0,0), C22 (1,1)
    U[0*2+0, 0] = 1 # A11
    U[1*2+1, 0] = 1 # A22
    V[0*2+0, 0] = 1 # B11
    V[1*2+1, 0] = 1 # B22
    W[0*2+0, 0] = 1 # C11
    W[1*2+1, 0] = 1 # C22
    
    # r=1: M2 = (A21+A22)B11
    # A21: i=1,j=0; A22: i=1,j=1
    # B11: j=0,k=0
    # C21: i=1,k=0
    U[1*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1 # C21: k=0, i=1 -> index 0*2+1=1
    
    # r=2: M3 = A11(B12-B22)
    # A11: i=0,j=0
    # B12: j=0,k=1; B22: j=1,k=1
    # C12: i=0,k=1
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1
    V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1 # C12: k=1, i=0 -> index 1*2+0=2
    
    # r=3: M4 = A22(B21-B11)
    # A22: i=1,j=1
    # B21: j=1,k=0; B11: j=0,k=0
    # C21: i=1,k=0; C11: i=0,k=0
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1
    V[0*2+0, 3] = -1
    W[0*2+1, 3] = 1 # C21
    W[0*2+0, 3] = 1 # C11
    
    # r=4: M5 = (A11+A12)B22
    # A11: i=0,j=0; A12: i=0,j=1
    # B22: j=1,k=1
    # C12: i=0,k=1; C22: i=1,k=1
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[1*2+0, 4] = 1 # C12
    W[1*2+1, 4] = 1 # C22
    
    # r=5: M6 = (A21-A11)(B11+B12)
    # A21: i=1,j=0; A11: i=0,j=0
    # B11: j=0,k=0; B12: j=0,k=1
    # C22: i=1,k=1
    U[1*2+0, 5] = 1
    U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1
    V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1 # C22
    
    # r=6: M7 = (A12-A22)(B21+B22)
    # A12: i=0,j=1; A22: i=1,j=1
    # B21: j=1,k=0; B22: j=1,k=1
    # C11: i=0,k=0; C22: i=1,k=1
    U[0*2+1, 6] = 1
    U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1
    V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1 # C11
    W[1*2+1, 6] = 1 # C22
    
    return U, V, W

def _block_224():
    # 2x2x4
    # i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # Split k into {0,1} and {2,3}
    # Two 2x2x2 blocks
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    
    U = np.zeros((4, 14))
    V = np.zeros((8, 14)) # j in {0,1}, k in {0..3} -> j*4+k
    W = np.zeros((8, 14)) # k in {0..3}, i in {0,1} -> k*2+i
    
    # Block 1: k in {0,1}
    # V index: j*4+k
    # W index: k*2+i
    for r in range(7):
        U[:, r] = U1[:, r]
        # V1 has j*2+k. We need j*4+k.
        for j in range(2):
            for k in range(2):
                V[j*4+k, r] = V1[j*2+k, r]
        # W1 has k*2+i. We need k*2+i. Same.
        for k in range(2):
            for i in range(2):
                W[k*2+i, r] = W1[k*2+i, r]
                
    # Block 2: k in {2,3}
    # Shift k by 2
    for r in range(7):
        U[:, r+7] = U2[:, r]
        for j in range(2):
            for k in range(2):
                V[j*4+(k+2), r+7] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2+i, r+7] = W2[k*2+i, r]
                
    return U, V, W

def _block_242():
    # 2x4x2
    # i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # Split j into {0,1} and {2,3}
    # Two 2x2x2 blocks
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    
    U = np.zeros((8, 14)) # i*4+j
    V = np.zeros((8, 14)) # j*2+k
    W = np.zeros((4, 14)) # k*2+i
    
    # Block 1: j in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+j, r] = U1[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V[j*2+k, r] = V1[j*2+k, r]
        W[:, r] = W1[:, r]
        
    # Block 2: j in {2,3}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2), r+7] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V[(j+2)*2+k, r+7] = V2[j*2+k, r]
        W[:, r+7] = W2[:, r]
        
    return U, V, W

def _block_244():
    # 2x4x4
    # i in {0,1}, j in {0..3}, k in {0..3}
    # Split j into {0,1} and {2,3}
    # Two 2x2x4 blocks
    U1, V1, W1 = _block_224()
    U2, V2, W2 = _block_224()
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28)) # j*4+k
    W = np.zeros((8, 28)) # k*2+i
    
    # Block 1: j in {0,1}
    for r in range(14):
        for i in range(2):
            for j in range(2):
                U[i*4+j, r] = U1[i*2+j, r]
        for j in range(2):
            for k in range(4):
                V[j*4+k, r] = V1[j*4+k, r]
        W[:, r] = W1[:, r]
        
    # Block 2: j in {2,3}
    for r in range(14):
        for i in range(2):
            for j in range(2):
                U[i*4+(j+2), r+14] = U2[i*2+j, r]
        for j in range(2):
            for k in range(4):
                V[(j+2)*4+k, r+14] = V2[j*4+k, r]
        W[:, r+14] = W2[:, r]
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # 2x4x6
    # Split p into 4 and 2
    # Block 1: 2x4x4
    # Block 2: 2x4x2
    
    U1, V1, W1 = _block_244()
    U2, V2, W2 = _block_242()
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42)) # j*6+k
    W = np.zeros((12, 42)) # k*2+i
    
    # Block 1: k in {0..3}
    for r in range(28):
        U[:, r] = U1[:, r]
        for j in range(4):
            for k in range(4):
                V[j*6+k, r] = V1[j*4+k, r]
        for k in range(4):
            for i in range(2):
                W[k*2+i, r] = W1[k*2+i, r]
                
    # Block 2: k in {4,5}
    for r in range(14):
        U[:, r+28] = U2[:, r]
        for j in range(4):
            for k in range(2):
                V[j*6+(k+4), r+28] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[(k+4)*2+i, r+28] = W2[k*2+i, r]
                
    return {"U": U, "V": V, "W": W}
