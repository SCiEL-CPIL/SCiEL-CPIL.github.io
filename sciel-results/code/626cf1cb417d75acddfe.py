def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 decomposition
    U_strassen = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A11
        [0, 0, 0, 0, 1, 1, -1], # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, -1, -1] # A22
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1]   # B22
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ])
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # U: 16 rows, V: 28 rows (but only first 16 relevant for k<4), W: 28 rows (but only first 16 relevant for k<4)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((28, 49))
    W_444 = np.zeros((28, 49))
    
    # Block index mappings for 2x2 blocks within 4x4
    # For A (and similarly B, C):
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # U indexing: row = i*4 + j
    # A11: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A12: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # A21: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # A22: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    # V indexing: row = j*7 + k
    # B11: (0,0)=0, (0,1)=1, (1,0)=7, (1,1)=8
    # B12: (0,2)=2, (0,3)=3, (1,2)=9, (1,3)=10
    # B21: (2,0)=14, (2,1)=15, (3,0)=21, (3,1)=22
    # B22: (2,2)=16, (2,3)=17, (3,2)=23, (3,3)=24
    
    # W indexing: row = k*4 + i
    # C11: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # C12: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # C21: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # C22: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    U_blocks = [
        [0, 1, 4, 5],   # A11: (i,j) pairs
        [2, 3, 6, 7],   # A12
        [8, 9, 12, 13], # A21
        [10, 11, 14, 15] # A22
    ]
    
    V_blocks = [
        [0, 1, 7, 8],   # B11: (j,k) pairs -> j*7+k
        [2, 3, 9, 10],  # B12
        [14, 15, 21, 22], # B21
        [16, 17, 23, 24]  # B22
    ]
    
    W_blocks = [
        [0, 1, 4, 5],   # C11: (k,i) pairs -> k*4+i
        [2, 3, 6, 7],   # C12
        [8, 9, 12, 13], # C21
        [10, 11, 14, 15] # C22
    ]
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            for b in range(4):
                coeff_top_U = U_strassen[b, r1]
                if coeff_top_U != 0:
                    for local_idx, flat_idx in enumerate(U_blocks[b]):
                        U_444[flat_idx, col] = coeff_top_U * U_strassen[local_idx, r2]
                
                coeff_top_V = V_strassen[b, r1]
                if coeff_top_V != 0:
                    for local_idx, flat_idx in enumerate(V_blocks[b]):
                        V_444[flat_idx, col] = coeff_top_V * V_strassen[local_idx, r2]
                
                coeff_top_W = W_strassen[b, r1]
                if coeff_top_W != 0:
                    for local_idx, flat_idx in enumerate(W_blocks[b]):
                        W_444[flat_idx, col] = coeff_top_W * W_strassen[local_idx, r2]
    
    # Verify 4x4x4 Strassen
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = np.dot(U_444[i*4+j], V_444[j*7+k] * W_444[k*4+i])
                assert abs(s - 1.0) < 1e-9, f"444 verify failed at {i},{j},{k}: {s}"
    
    # Build 4x4x3 schoolbook decomposition (rank 48)
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((28, 48))
    W_443 = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_full = k + 4  # k in [4,7)
                U_443[i*4+j, col] = 1
                V_443[j*7+k_full, col] = 1
                W_443[k_full*4+i, col] = 1
                col += 1
    
    # Verify 4x4x3 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_full = k + 4
                s = np.dot(U_443[i*4+j], V_443[j*7+k_full] * W_443[k_full*4+i])
                assert abs(s - 1.0) < 1e-9, f"443 verify failed at {i},{j},{k}: {s}"
    
    # Combine
    total_rank = 49 + 48
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    U[:, :49] = U_444
    U[:, 49:] = U_443
    
    V[:, :49] = V_444
    V[:, 49:] = V_443
    
    W[:, :49] = W_444
    W[:, 49:] = W_443
    
    # Full verification
    for i in range(4):
        for j in range(4):
            for k in range(7):
                s = np.dot(U[i*4+j], V[j*7+k] * W[k*4+i])
                assert abs(s - 1.0) < 1e-9, f"Full verify failed at {i},{j},{k}: {s}"
    
    # Convert to list
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
