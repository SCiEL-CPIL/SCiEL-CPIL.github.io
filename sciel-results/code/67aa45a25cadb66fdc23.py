import numpy as np
import time

DIMENSIONS = [4, 4, 8]
TIME_LIMIT = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Use Kronecker product to build 4x4x4 (rank 49)
    # U4[i*4+j, ro*7+ri] = U2[i*2+io, ro] * U2[j*2+jo, ri]
    # where i,j in 0..1, io,jo in 0..1
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)
    
    # Verify shapes
    assert U4.shape == (16, 49)
    assert V4.shape == (16, 49)
    assert W4.shape == (16, 49)
    
    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of output)
    U[:, 0:49] = U4
    # V4 has shape (16, 49), indexed by [j*4+k, r]
    # For 4x4x8, V is indexed by [j*8+k, r] where j in 0..3, k in 0..7
    # For left 4 columns: k in 0..3
    for k in range(4):
        V[k::4, 0:49] = V4[:, :]  # This is wrong, let me think again
    # Actually V4[j*4+k, r] should map to V[j*8+k, r] for k=0..3
    V[:, 0:49] = 0
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
    
    # W4[k*4+i, r] should map to W[k*4+i, r] for k=0..3 (left block)
    W[:, 0:49] = W4
    
    # Right block (columns 4-7 of output)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(4+k), 49:98] = V4[j*4+k, :]
    W[:, 49:98] = W4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
