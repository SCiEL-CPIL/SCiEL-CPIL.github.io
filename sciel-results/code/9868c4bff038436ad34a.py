import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 150.0  # 150s wall-clock limit, use 145s for safety
    
    # We will construct the 4x4 Strassen decomposition using Kronecker product of 2x2 Strassen.
    
    # 1. Construct 2x2 Strassen factors
    # U2 shape: (4, 7) where row idx = i*2 + j for A[i,j]
    # V2 shape: (4, 7) where row idx = j*2 + k for B[j,k]
    # W2 shape: (4, 7) where row idx = k*2 + i for C[k,i]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A00 + A11) * (B00 + B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: M2 = (A10 + A11) * B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # Term 2: M3 = A00 * (B01 - B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 3: M4 = A11 * (B10 - B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 4: M5 = (A00 + A01) * B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 5: M6 = (A10 - A00) * (B00 + B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: M7 = (A01 - A11) * (B10 + B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # 2. Compute 4x4 factors via Kronecker product
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    # The Kronecker product of the tensors corresponds to the tensor decomposition of the Kronecker product of the matrices.
    # Indexing check:
    # A4 index (I, i', J, j') -> flat (I*4 + J)*? No, flat is (I*2+i')*4 + (J*2+j').
    # Kronecker product (U2 ⊗ U2) row index (2*I + i', 2*J + j') -> flat (2*I+i')*4 + (2*J+j').
    # This matches the row-major flattening of the 4x4 matrix.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to lists of lists with integer values
    U4 = U4.tolist()
    V4 = V4.tolist()
    W4 = W4.tolist()
    
    # Ensure exact integers
    for i in range(len(U4)):
        for j in range(len(U4[0])):
            U4[i][j] = int(U4[i][j])
    for i in range(len(V4)):
        for j in range(len(V4[0])):
            V4[i][j] = int(V4[i][j])
    for i in range(len(W4)):
        for j in range(len(W4[0])):
            W4[i][j] = int(W4[i][j])
            
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
