import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    
    # Identity decomposition for 4x4 matrix multiplication
    # Rank = 4*4*4 = 64
    # U[i*4+j, r] = 1 if r == i*4+j else 0
    # V[j*4+k, r] = 1 if r == j*4+k else 0
    # W[k*4+i, r] = 1 if r == k*4+i else 0
    # Wait, the tensor is T[i,j,k] = delta_{i,j} delta_{j,k}? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Term r corresponds to (i,j,k).
    # U[i,j, r] = 1 if r encodes (i,j) AND k matches?
    # No, in the decomposition:
    # C[i,k] = sum_r U[i,j, r] V[j,k, r] W[k,i, r]?
    # The interface is:
    # U shape (n*m, rank) -> (16, rank). Row i*4+j.
    # V shape (m*p, rank) -> (16, rank). Row j*4+k.
    # W shape (p*n, rank) -> (16, rank). Row k*4+i.
    
    # For each term r (0..63), let r = i*16 + j*4 + k.
    # U[i*4+j, r] = 1
    # V[j*4+k, r] = 1
    # W[k*4+i, r] = 1
    
    rank = 64
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                r = i * 16 + j * 4 + k
                U[i * 4 + j, r] = 1
                V[j * 4 + k, r] = 1
                W[k * 4 + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
