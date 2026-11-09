import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Part 1: M_{4,4,4} rank 49
    # Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Kronecker product for 4x4
    U1 = np.kron(U2, U2) # 16x49
    V1 = np.kron(V2, V2) # 16x49
    W1 = np.kron(W2, W2) # 16x49
    
    # Part 2: M_{4,4,1} rank 16
    # A_L (4x4) * b (4x1)
    # U is identity 4x4 x 4x1?
    # U2_part is 16x16. U2_part[i*4+j, k*4+l] = delta_{i,k} delta_{j,l}?
    # No, M_{4,4,1} computes C_i = sum_j A_{ij} b_j.
    # Tensor T_{i,j,k} = delta_{j,k}.
    # U_{i,j, r} V_{k, r} W_{i, r}
    # r = j.
    # U_{i,j, j} = 1.
    # V_{j, j} = 1.
    # W_{i, j} = 1.
    U2_part = np.zeros((16, 16))
    V2_part = np.zeros((4, 16))
    W2_part = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            r = j
            U2_part[i*4 + j, r] = 1
            V2_part[j, r] = 1
            W2_part[i, r] = 1
            
    # Part 3: M_{4,1,4} rank 16
    # a (4x1) * b (1x4)
    # C_{i,k} = a_i b_k.
    # Tensor T_{i,j,k,l} = delta_{j,0} delta_{l,0}?
    # a is col 4 of A. b is row 4 of B.
    # U_{i, 4, r} V_{4, k, r} W_{i, k, r}
    # r = i*4+k? No, rank 16.
    # r = i*4+k is 16 values.
    # U_{i,4, i*4+k} = 1?
    # Then sum_r U V W = sum_{i,k} a_i b_k delta_{i,i} delta_{k,k}?
    # No, we need C_{i,k} = a_i b_k.
    # Let r = i*4+k.
    # U_{i,4, r} = 1 if r == i*4+k?
    # V_{4,k, r} = 1 if r == i*4+k?
    # W_{i,k, r} = 1 if r == i*4+k?
    # Then term is a_i b_k. Sum is sum_{i,k} a_i b_k = (sum a_i)(sum b_k). Wrong.
    # We need C_{i,k} = a_i b_k.
    # This is the outer product.
    # Rank of outer product is 1?
    # No, C is 4x4.
    # C_{i,k} = a_i b_k.
    # This is rank 1 matrix.
    # But the tensor rank?
    # T_{i, j, k, l} with j=4, l=4.
    # T_{i, 4, k, 4} = 1.
    # This tensor is sum_{i,k} e_i \otimes e_4 \otimes e_k \otimes e_4.
    # This is not rank 1.
    # It is the direct sum of 16 rank 1 tensors?
    # Yes, for each (i,k), we have a term a_i b_k.
    # So rank is 16.
    # U_{i,4, i*4+k} = 1.
    # V_{4,k, i*4+k} = 1.
    # W_{i,k, i*4+k} = 1.
    U3_part = np.zeros((4, 16))
    V3_part = np.zeros((4, 16))
    W3_part = np.zeros((16, 16))
    for i in range(4):
        for k in range(4):
            r = i*4 + k
            U3_part[i, r] = 1
            V3_part[k, r] = 1
            W3_part[k*4 + i, r] = 1 # W index is k*n+i
            
    # Part 4: M_{4,1,1} rank 1
    # a (4x1) * b (1x1)
    # C_i = a_i b.
    # Rank 1.
    U4_part = np.zeros((4, 1))
    V4_part = np.zeros((1, 1))
    W4_part = np.zeros((4, 1))
    for i in range(4):
        U4_part[i, 0] = 1
        V4_part[0, 0] = 1
        W4_part[i, 0] = 1
        
    # Assemble
    # U: 20x82
    # V: 25x82
    # W: 20x82
    
    U = np.zeros((20, 82))
    V = np.zeros((25, 82))
    W = np.zeros((20, 82))
    
    col = 0
    # Part 1
    # U1 is 16x49. Maps to A[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col:col+49] = U1[i*4 + j, :]
    # V1 is 16x49. Maps to B[0:4, 0:4]
    for j in range(4):
        for k in range(4):
            V[j*5 + k, col:col+49] = V1[j*4 + k, :]
    # W1 is 16x49. Maps to C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            W[k*4 + i, col:col+49] = W1[k*4 + i, :]
    col += 49
            
    # Part 2
    # U2_part is 16x16. Maps to A[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col:col+16] = U2_part[i*4 + j, :]
    # V2_part is 4x16. Maps to B[0:4, 4]
    for j in range(4):
        V[j*5 + 4, col:col+16] = V2_part[j, :]
    # W2_part is 4x16. Maps to C[0:4, 4]
    for i in range(4):
        W[4*4 + i, col:col+16] = W2_part[i, :]
    col += 16
        
    # Part 3
    # U3_part is 4x16. Maps to A[0:4, 4]
    for i in range(4):
        U[i*5 + 4, col:col+16] = U3_part[i, :]
    # V3_part is 4x16. Maps to B[4, 0:4]
    for k in range(4):
        V[4*5 + k, col:col+16] = V3_part[k, :]
    # W3_part is 16x16. Maps to C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            W[k*4 + i, col:col+16] = W3_part[k*4 + i, :]
    col += 16
        
    # Part 4
    # U4_part is 4x1. Maps to A[0:4, 4]
    for i in range(4):
        U[i*5 + 4, col:col+1] = U4_part[i, :]
    # V4_part is 1x1. Maps to B[4, 4]
    V[4*5 + 4, col:col+1] = V4_part[0, :]
    # W4_part is 4x1. Maps to C[0:4, 4]
    for i in range(4):
        W[4*4 + i, col:col+1] = W4_part[i, :]
    col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
