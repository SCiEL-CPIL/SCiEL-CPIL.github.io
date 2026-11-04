import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Construct 4x4x4 factors
    # Rank 49
    U44 = np.zeros((16, 49), dtype=float)
    V44 = np.zeros((16, 49), dtype=float)
    W44 = np.zeros((16, 49), dtype=float)
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # Calculate effective coefficients for blocks
            # Coeff for A block k
            # c_A[k] = sum_u U22[u, s] * U22[k, r]
            c_A = np.dot(U22[:, s], U22[:, r])
            # Coeff for B block l
            c_B = np.dot(V22[:, s], V22[:, r])
            # Coeff for C block w
            c_C = W22[:, r] * W22[:, s]
            
            # Expand to 4x4 indices
            for k in range(4):
                if c_A[k] != 0:
                    # Block k corresponds to indices in 4x4
                    # k=0: 0,1,4,5
                    # k=1: 2,3,6,7
                    # k=2: 8,9,12,13
                    # k=3: 10,11,14,15
                    # Mapping: k -> (i_block, j_block)
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    i_b = k // 2
                    j_b = k % 2
                    for i in range(2):
                        for j in range(2):
                            idx = (i_b * 2 + i) * 4 + (j_b * 2 + j)
                            U44[idx, col] = c_A[k]
                            
            for l in range(4):
                if c_B[l] != 0:
                    i_b = l // 2
                    j_b = l % 2
                    for i in range(2):
                        for j in range(2):
                            idx = (i_b * 2 + i) * 4 + (j_b * 2 + j)
                            V44[idx, col] = c_B[l]
                            
            for w in range(4):
                if c_C[w] != 0:
                    i_b = w // 2
                    j_b = w % 2
                    for i in range(2):
                        for j in range(2):
                            idx = (i_b * 2 + i) * 4 + (j_b * 2 + j)
                            W44[idx, col] = c_C[w]
                            
    # Construct 4x4x9 factors
    # Rank 114
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((36, total_rank), dtype=float)
    W = np.zeros((36, total_rank), dtype=float)
    
    # Part 1: B cols 0-3
    U[:, 0:49] = U44
    # V mapping: V44 idx (j*4+k) -> V idx (j*9+k)
    for j in range(4):
        for k in range(4):
            v44_idx = j * 4 + k
            v9_idx = j * 9 + k
            V[v9_idx, 0:49] = V44[v44_idx, 0:49]
            
    # W mapping: W44 idx (k*4+i) -> W idx (k*9+i) ?
    # W convention: W[k*n+i, r]. n=4.
    # W44 is for 4x4x4. p=4.
    # W44 idx is k*4+i.
    # W9 idx is k*4+i. (Since n=4, the index into W is k*n+i regardless of p?
    # Wait. W shape is (p*n, rank).
    # For 4x4x4, p=4, n=4. W size 16. Index k*4+i.
    # For 4x4x9, p=9, n=4. W size 36. Index k*4+i.
    # The k index corresponds to the column of B / column of C.
    # So for Part 1, k is 0..3.
    # So W[0:16, 0:49] = W44.
    W[0:16, 0:49] = W44
    
    # Part 2: B cols 4-7
    U[:, 49:98] = U44
    # V mapping: V44 idx (j*4+k_local) -> V idx (j*9 + 4 + k_local)
    for j in range(4):
        for k_local in range(4):
            v44_idx = j * 4 + k_local
            v9_idx = j * 9 + (4 + k_local)
            V[v9_idx, 49:98] = V44[v44_idx, 0:49]
            
    # W mapping: k_global = 4 + k_local.
    # W idx = k_global * 4 + i.
    for k_local in range(4):
        k_global = 4 + k_local
        for i in range(4):
            w44_idx = k_local * 4 + i
            w9_idx = k_global * 4 + i
            W[w9_idx, 49:98] = W44[w44_idx, 0:49]
            
    # Part 3: B col 8
    col_offset = 98
    for i in range(4):
        for j in range(4):
            c = col_offset + i * 4 + j
            U[i * 4 + j, c] = 1.0
            V[j * 9 + 8, c] = 1.0
            W[8 * 4 + i, c] = 1.0
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
