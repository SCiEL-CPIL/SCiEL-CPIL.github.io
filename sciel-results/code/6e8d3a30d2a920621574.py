import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2 factor matrices
    # U2 rows: (i,j) -> i*2+j
    # V2 rows: (j,k) -> j*2+k
    # W2 rows: (k,i) -> k*2+i
    
    U2 = [
        [1,  0,  1,  0,  1, -1,  0], # 00
        [0,  0,  0,  0,  1,  0,  1], # 01
        [0,  1,  0,  0,  0,  1,  0], # 10
        [1,  1,  0,  1,  0,  0, -1]  # 11
    ]
    
    V2 = [
        [1,  1,  0, -1,  0,  1,  0], # 00
        [0,  0,  1,  0,  0,  1,  0], # 01
        [0,  0,  0,  1,  0,  0,  1], # 10
        [1,  0, -1,  0,  1,  0,  1]  # 11
    ]
    
    W2 = [
        [1,  0,  0,  0,  1,  0,  1], # 00 (C00)
        [0,  0,  1,  0, -1,  0,  0], # 01 (C01) -> k=0, i=1
        [0,  1,  0,  1,  0,  0,  0], # 10 (C10) -> k=1, i=0
        [1, -1,  1, -1,  0,  1,  0]  # 11 (C11) -> k=1, i=1
    ]
    
    # Convert to numpy for easy Kronecker
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    # Kronecker product
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # Convert back to list of lists
    U4 = U4_np.tolist()
    V4 = V4_np.tolist()
    W4 = W4_np.tolist()
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
