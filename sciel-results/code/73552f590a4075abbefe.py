import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication
    # U: (i,j) -> index i*2+j
    # V: (j,k) -> index j*2+k
    # W: (k,i) -> index k*2+i (Trace convention)
    
    # 7 columns for rank 7
    # Rows for U, V are 4 (2x2 inputs)
    # Rows for W are 4 (2x2 outputs)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a (00)
        [0, 0, 0, 0, 1, 0, 1],   # b (01)
        [0, 1, 0, 0, 0, 1, 0],   # c (10)
        [1, 1, 0, 1, 0, 0, -1]   # d (11)
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # e (00)
        [0, 0, 1, 0, 0, 1, 0],   # f (01)
        [0, 0, 0, 1, 0, 0, 1],   # g (10)
        [1, 0, -1, 0, 1, 0, 1]   # h (11)
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (k=1, i=1)
    ], dtype=int)
    
    # Kronecker product to get 4x4 decomposition
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to list of lists as required
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
