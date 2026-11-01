import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 7 blocks of (2,2,2) and 1 block of (2,2,1)
    # Rank(2,2,2) = 7, Rank(2,2,1) = 4
    rank = 7 * 7 + 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U vectors (A space, dim 4: 11, 12, 21, 22)
    # V vectors (B space, dim 4: 11, 12, 21, 22)
    # W vectors (C space, dim 4: 11, 12, 21, 22)
    
    # M1
    U[:, col] = [1, 0, 0, 1]
    V[:, col] = [1, 0, 0, 1] # Wait, B indices for block are local
    # We need to map local B indices to global V indices
    # Local B: 11, 12, 21, 22
    # Global B for block at k_start: 
    # B[0, k_start], B[1, k_start], B[0, k_start+1], B[1, k_start+1]
    
    # It is easier to define the local vectors and then scatter them.
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [0, 0, 1, 0], # M4
        [0, -1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Wait, W is C space.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So W vectors for each M:
    # M1: [1, 0, 0, 1] (C11, C22)
    # M2: [0, 0, 1, -1] (C21, -C22)
    # M3: [0, 1, 0, 1] (C12, C22)
    # M4: [1, 0, 1, 0] (C11, C21)
    # M5: [-1, 1, 0, 0] (-C11, C12)
    # M6: [0, 0, 0, 1] (C22)
    # M7: [1, 0, 0, 0] (C11)
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for block_idx in range(7):
        k_start = block_idx * 2
        for s in range(7):
            # U
            U[:, col] = strassen_U[s]
            
            # V: Map local B indices [11, 12, 21, 22] to global
            # Local 11 -> Global [0, k_start] -> index 0*15 + k_start? No.
            # V shape is (m*p, rank). m=2, p=15.
            # V[j*p + k, col].
            # Local B is 2x2.
            # Local (0,0) -> Global (0, k_start)
            # Local (0,1) -> Global (0, k_start+1)
            # Local (1,0) -> Global (1, k_start)
            # Local (1,1) -> Global (1, k_start+1)
            
            v_vec = strassen_V[s]
            V[0*p + k_start, col] = v_vec[0]
            V[0*p + k_start+1, col] = v_vec[1]
            V[1*p + k_start, col] = v_vec[2]
            V[1*p + k_start+1, col] = v_vec[3]
            
            # W: Map local C indices [11, 12, 21, 22] to global
            # W shape is (p*n, rank). p=15, n=2.
            # W[k*n + i, col].
            # Local (0,0) -> Global (k_start, 0) -> index k_start*2 + 0
            # Local (0,1) -> Global (k_start+1, 0) -> index (k_start+1)*2 + 0
            # Local (1,0) -> Global (k_start, 1) -> index k_start*2 + 1
            # Local (1,1) -> Global (k_start+1, 1) -> index (k_start+1)*2 + 1
            
            w_vec = strassen_W[s]
            W[k_start*n + 0, col] = w_vec[0]
            W[(k_start+1)*n + 0, col] = w_vec[1]
            W[k_start*n + 1, col] = w_vec[2]
            W[(k_start+1)*n + 1, col] = w_vec[3]
            
            col += 1
            
    # Last block: 2x2x1 at k_start = 14
    k_start = 14
    # Schoolbook for 2x2x1
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # B is 2x1. Local B indices: 11, 21. (j=0,1; k=0)
    # Local C indices: 1, 2. (i=0,1)
    
    # M1: A11 B1 -> C1
    U[:, col] = [1, 0, 0, 0]
    V[0*p + k_start, col] = 1
    V[1*p + k_start, col] = 0
    W[k_start*n + 0, col] = 1
    W[k_start*n + 1, col] = 0
    col += 1
    
    # M2: A12 B2 -> C1
    U[:, col] = [0, 1, 0, 0]
    V[0*p + k_start, col] = 0
    V[1*p + k_start, col] = 1
    W[k_start*n + 0, col] = 1
    W[k_start*n + 1, col] = 0
    col += 1
    
    # M3: A21 B1 -> C2
    U[:, col] = [0, 0, 1, 0]
    V[0*p + k_start, col] = 1
    V[1*p + k_start, col] = 0
    W[k_start*n + 0, col] = 0
    W[k_start*n + 1, col] = 1
    col += 1
    
    # M4: A22 B2 -> C2
    U[:, col] = [0, 0, 0, 1]
    V[0*p + k_start, col] = 0
    V[1*p + k_start, col] = 1
    W[k_start*n + 0, col] = 0
    W[k_start*n + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
