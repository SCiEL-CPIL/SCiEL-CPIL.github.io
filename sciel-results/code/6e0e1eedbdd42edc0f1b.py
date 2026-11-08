import numpy as np

def construct():
    n, m, p = 3, 3, 6
    
    # We decompose the problem into two 3x3x3 matrix multiplications.
    # C[:, 0:3] = A * B[:, 0:3]
    # C[:, 3:6] = A * B[:, 3:6]
    
    # Each 3x3x3 multiplication has a trivial rank of 9.
    # Total rank = 9 + 9 = 18.
    
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((n * p, rank))
    
    # Block 1: B columns 0,1,2 -> C columns 0,1,2
    # Indices in B: j*p + k where j in 0..2, k in 0..2
    # Indices in C: i*p + k where i in 0..2, k in 0..2
    # Indices in A: i*m + j where i in 0..2, j in 0..2
    
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is local index 0,1,2
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                
                U[a_idx, col_offset] = 1.0
                V[b_idx, col_offset] = 1.0
                W[c_idx, col_offset] = 1.0
                col_offset += 1
                
    # Block 2: B columns 3,4,5 -> C columns 3,4,5
    # Indices in B: j*p + k where j in 0..2, k in 3..5
    # Indices in C: i*p + k where i in 0..2, k in 3..5
    
    for i in range(n):
        for j in range(m):
            for k in range(3, 6): # k is global index 3,4,5
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                
                U[a_idx, col_offset] = 1.0
                V[b_idx, col_offset] = 1.0
                W[c_idx, col_offset] = 1.0
                col_offset += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
