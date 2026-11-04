import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Base Strassen 2x2 matrices
    # U2 maps A (4 elements) to 7 products
    # V2 maps B (4 elements) to 7 products
    # W2 maps 7 products to C (4 elements)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    # Kronecker products for 4x4
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # W4_np rows are in row-major order of C (index i*4+k for C[i,k])
    # Required W index is k*4+i for C[i,k] (column-major)
    n = 4
    W_permuted = np.zeros_like(W4_np)
    for i in range(n):
        for k in range(n):
            new_idx = k * n + i
            old_idx = i * n + k
            W_permuted[new_idx] = W4_np[old_idx]
            
    # Convert to lists
    U = U4_np.tolist()
    V = V4_np.tolist()
    W = W_permuted.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
