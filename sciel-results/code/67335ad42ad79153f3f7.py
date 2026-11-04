import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, -1, 0],
        [0, 1, 0, -1]
    ]).T # Shape (4, 7)
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]).T # Shape (4, 7)
    
    W2 = np.array([
        [1, 0, 0, 1, 0, 0, -1],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, -1, -1, 1, 0]
    ]) # Shape (4, 7)
    
    # Kronecker product for 4x4x4
    # U4 shape (16, 49)
    U4 = np.kron(U2, U2)
    # V4 shape (16, 49)
    V4 = np.kron(V2, V2)
    # W4 shape (16, 49)
    W4 = np.kron(W2, W2)
    
    # Construct 4x4x8 factors
    rank = 98
    U_total = np.zeros((n*m, rank))
    V_total = np.zeros((m*p, rank))
    W_total = np.zeros((p*n, rank))
    
    # Left half (k=0..3)
    U_total[:, 0:49] = U4
    
    # V4 rows are indexed by j*4+k (j=0..3, k=0..3)
    # V_total rows are indexed by j*8+k (j=0..3, k=0..7)
    # For left half, we use k=0..3
    for j in range(4):
        for k in range(4):
            v4_idx = j*4 + k
            v_total_idx = j*8 + k
            V_total[v_total_idx, 0:49] = V4[v4_idx, :]
            
    # W4 rows are indexed by k*4+i (k=0..3, i=0..3)
    # W_total rows are indexed by k*4+i (k=0..7, i=0..3)
    # For left half, we use k=0..3
    for k in range(4):
        for i in range(4):
            w4_idx = k*4 + i
            w_total_idx = k*4 + i
            W_total[w_total_idx, 0:49] = W4[w4_idx, :]
            
    # Right half (k=4..7)
    U_total[:, 49:98] = U4
    
    # For right half, V uses k=4..7
    for j in range(4):
        for k in range(4):
            v4_idx = j*4 + k
            v_total_idx = j*8 + (k+4)
            V_total[v_total_idx, 49:98] = V4[v4_idx, :]
            
    # For right half, W uses k=4..7
    for k in range(4):
        for i in range(4):
            w4_idx = k*4 + i
            w_total_idx = (k+4)*4 + i
            W_total[w_total_idx, 49:98] = W4[w4_idx, :]
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
