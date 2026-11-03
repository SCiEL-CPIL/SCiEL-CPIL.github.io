import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: Decompose T[i, j, k] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Part 1: 4x4x4 sub-block (i, j, k in 0..3)
    # Use Strassen-like rank 49 decomposition for 4x4x4
    # Construct 2x2x2 rank 7 tensor
    U7 = np.zeros((4, 7))
    V7 = np.zeros((4, 7))
    W7 = np.zeros((4, 7))
    
    # Strassen indices for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U7 rows: 00, 01, 10, 11 (i, j)
    # V7 rows: 00, 01, 10, 11 (j, k)
    # W7 rows: 00, 01, 10, 11 (k, i)
    
    # Col 0: M1
    U7[0, 0] = 1; U7[3, 0] = 1
    V7[0, 0] = 1; V7[3, 0] = 1
    W7[0, 0] = 1; W7[3, 0] = 1; W7[1, 0] = 1; W7[2, 0] = 1 # W contributes to C11, C12, C21, C22?
    # Wait, W index is (k, i).
    # C11 (0,0) gets M1. C12 (0,1) gets M1. C21 (1,0) gets M1. C22 (1,1) gets M1.
    # So W7[0,0]=1 (k=0,i=0), W7[1,0]=1 (k=1,i=0), W7[2,0]=1 (k=0,i=1), W7[3,0]=1 (k=1,i=1).
    
    # Col 1: M2
    U7[2, 1] = 1; U7[3, 1] = 1 # A21, A22
    V7[0, 1] = 1 # B11
    W7[2, 1] = 1 # C21 (k=0, i=1)
    
    # Col 2: M3
    U7[0, 2] = 1 # A11
    V7[1, 2] = 1; V7[3, 2] = -1 # B12 - B22
    W7[1, 2] = 1 # C12 (k=1, i=0)
    
    # Col 3: M4
    U7[3, 3] = 1 # A22
    V7[2, 3] = 1; V7[0, 3] = -1 # B21 - B11
    W7[2, 3] = 1 # C21 (k=0, i=1)
    
    # Col 4: M5
    U7[0, 4] = 1; U7[1, 4] = 1 # A11, A12
    V7[3, 4] = 1 # B22
    W7[1, 4] = 1; W7[3, 4] = 1 # C12, C22
    
    # Col 5: M6
    U7[2, 5] = 1; U7[0, 5] = -1 # A21 - A11
    V7[0, 5] = 1; V7[1, 5] = 1 # B11 + B12
    W7[0, 5] = 1; W7[2, 5] = 1 # C11, C21
    
    # Col 6: M7
    U7[1, 6] = 1; U7[3, 6] = -1 # A12 - A22
    V7[2, 6] = 1; V7[3, 6] = 1 # B21 + B22
    W7[0, 6] = 1; W7[1, 6] = 1 # C11, C12
    
    # Construct 4x4x4 rank 49 via Kronecker product
    U44 = np.kron(U7, U7) # 16x49
    V44 = np.kron(V7, V7) # 16x49
    W44 = np.kron(W7, W7) # 16x49
    
    # Initialize full matrices
    rank = 49 + 16 + 16 + 4
    U = np.zeros((20, rank))
    V = np.zeros((25, rank))
    W = np.zeros((20, rank))
    
    # Fill Part 1: i, j, k in 0..3
    # U indices: i*m + j = i*5 + j. For i,j in 0..3, indices are 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18.
    # V indices: j*p + k = j*5 + k. For j,k in 0..3, indices are 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18.
    # W indices: k*n + i = k*4 + i. For k,i in 0..3, indices are 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15.
    
    # Map 4x4 indices to 5x5 indices
    # U44 row (u, v) where u,v in 0..3. Index in U44 is u*4+v? No, U7 is 2x2, so U44 is 4x4.
    # U7 rows are 00, 01, 10, 11.
    # U44 rows are (00,00), (00,01), (00,10), (00,11), (01,00)...
    # This corresponds to i, j in binary?
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # U44 row index r corresponds to (i1, i2, j1, j2).
    # We need to map this to the linear index i*5 + j.
    
    u_map = []
    v_map = []
    w_map = []
    for i1 in range(2):
        for i2 in range(2):
            i = i1 * 2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    u_map.append(i * 5 + j)
                    
    for j1 in range(2):
        for j2 in range(2):
            j = j1 * 2 + j2
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    v_map.append(j * 5 + k)
                    
    for k1 in range(2):
        for k2 in range(2):
            k = k1 * 2 + k2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    w_map.append(k * 4 + i)

    # Fill U, V, W for Part 1
    for c in range(49):
        for r in range(16):
            U[u_map[r], c] = U44[r, c]
            V[v_map[r], c] = V44[r, c]
            W[w_map[r], c] = W44[r, c]

    # Part 2: A2 B3 (i in 0..3, j=4, k in 0..3)
    # U index: i*5 + 4
    # V index: 4*5 + k = 20 + k
    # W index: k*4 + i
    # Rank 16 (schoolbook)
    offset = 49
    for i in range(4):
        for k in range(4):
            col = offset + i * 4 + k
            U[i * 5 + 4, col] = 1
            V[20 + k, col] = 1
            W[k * 4 + i, col] = 1

    # Part 3: A1 B2 (i in 0..3, j in 0..3, k=4)
    # U index: i*5 + j
    # V index: j*5 + 4
    # W index: 4*4 + i = 16 + i
    # Rank 16
    offset = 49 + 16
    for i in range(4):
        for j in range(4):
            col = offset + i * 4 + j
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[16 + i, col] = 1

    # Part 4: A2 B4 (i in 0..3, j=4, k=4)
    # U index: i*5 + 4
    # V index: 4*5 + 4 = 24
    # W index: 4*4 + i = 16 + i
    # Rank 4
    offset = 49 + 16 + 16
    for i in range(4):
            col = offset + i
            U[i * 5 + 4, col] = 1
            V[24, col] = 1
            W[16 + i, col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
