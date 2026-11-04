import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # r=0: (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # A indices: 00(0), 01(1), 10(2), 11(3)
    # B indices: 00(0), 01(1), 10(2), 11(3)
    # C indices: 00(0), 01(1), 10(2), 11(3)
    
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # r=1: (A10+A11)B00 -> C10 += M1, C11 -= M1
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # r=2: A00(B01-B11) -> C01 += M2, C11 += M2
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # r=3: A11(B10-B00) -> C10 += M3, C00 += M3
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[2,3]=1; W2[0,3]=1
    
    # r=4: (A00+A01)B11 -> C00 -= M4, C01 += M4
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # r=5: (A10-A00)(B00+B01) -> C10 += M5
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[2,5]=1
    
    # r=6: (A01-A11)(B10+B11) -> C00 += M6
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    # Build 4x4x4 decomposition via Kronecker product
    # U4 = U2 (x) U2, V4 = V2 (x) V2, W4 = W2 (x) W2
    # Shape: (16, 49)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Reorder indices to match standard row-major flattening?
    # np.kron(A, B) for A (m1, n1) and B (m2, n2) gives (m1*m2, n1*n2).
    # Index (i1*m2 + i2, j1*n2 + j2) = A[i1, j1] * B[i2, j2].
    # This matches the index mapping i = 2*Io + Ii, j = 2*Jo + Ji.
    # So U4[i*4+j, r] is correct.
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First half (k in 0..3)
    # U[:, 0:49] = U4
    U[:, 0:49] = U4
    
    # V mapping for first half
    # V[8*j + k, r] for k in 0..3
    # V4 has index 4*j + k.
    # We can construct V[:, 0:49] by placing V4 in the first 4 cols of each row block of 8.
    # V shape 32x49.
    # Rows 0-7 correspond to j=0 (k=0..7).
    # V[0:4, 0:49] = V4[0:4, :]  (j=0, k=0..3)
    # V[4:8, 0:49] = 0            (j=0, k=4..7)
    # V[8:12, 0:49] = V4[4:8, :]  (j=1, k=0..3)
    # ...
    for j in range(4):
        V[j*8 : j*8+4, 0:49] = V4[j*4 : j*4+4, :]
        
    # W mapping for first half
    # W[4*k + i, r] for k in 0..3
    # W4 has index 4*k + i.
    # W[0:16, 0:49] = W4[:, :]
    # W[16:32, 0:49] = 0
    W[0:16, 0:49] = W4
    
    # Second half (k in 4..7)
    # U[:, 49:98] = U4
    U[:, 49:98] = U4
    
    # V mapping for second half
    # V[8*j + (4+k'), r] for k' in 0..3
    for j in range(4):
        V[j*8+4 : j*8+8, 49:98] = V4[j*4 : j*4+4, :]
        
    # W mapping for second half
    # W[4*k + i, r] for k in 4..7
    # Let k = 4+k'. Index 4*(4+k') + i = 16 + 4*k' + i.
    # W4 index is 4*k' + i.
    # So W[16:32, 49:98] = W4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
