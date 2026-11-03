"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-114 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Decompose the 4x4x9 problem into two 4x4x4 blocks and one 4x4x1 block.
    - Use the Kronecker product of the rank-7 Strassen decomposition for 2x2x2 
      to get a rank-49 decomposition for 4x4x4.
    - Use schoolbook (rank-16) for the 4x4x1 block.
    - Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # 1. Define rank-7 decomposition for 2x2x2 (Strassen)
    # U, V, W are 4x7 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for U, V
    # W indices: 0:C00, 1:C10, 2:C01, 3:C11 (k*n+i)
    
    U2 = np.array([
        [1,  0,  1,  0,  1,  1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0, -1, 0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ], dtype=float)
    
    V2 = np.array([
        [1,  1,  0, -1, 0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1, 0,  1,  0,  1]   # B11
    ], dtype=float)
    
    W2 = np.array([
        [1,  0,  0,  1, -1, 0,  1],  # C00
        [0,  1,  0,  1,  0,  0,  0],  # C10
        [0,  0,  1,  0,  1,  0,  0],  # C01
        [1, -1, 1,  0,  0,  1,  1]   # C11
    ], dtype=float)
    
    # 2. Kronecker product for 4x4x4
    U444 = np.kron(U2, U2)  # 16x49
    V444 = np.kron(V2, V2)  # 16x49
    W444 = np.kron(W2, W2)  # 16x49
    
    # 3. Schoolbook for 4x4x1
    # Rank 16
    U441 = np.zeros((16, 16))
    V441 = np.zeros((16, 16))
    W441 = np.zeros((16, 16))
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U441[i * 4 + j, col] = 1
            V441[j * 1 + 0, col] = 1  # B[j, 0]
            W441[0 * 4 + i, col] = 1   # C[0, i] (k=0, n=4)
            
    # 4. Assemble final matrices for 4x4x9
    # Total rank = 49 + 49 + 16 = 114
    total_rank = 114
    
    U = np.zeros((16, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((36, total_rank)) # p*n = 9*4 = 36
    
    # Block 1: Cols 0-3 of output (uses B cols 0-3)
    U[:, 0:49] = U444
    V[:, 0:49] = V444
    W[0:16, 0:49] = W444 # C cols 0-3 correspond to W rows 0-15 (k=0..3, n=4? No)
    
    # W convention: W[k*n + i, r]. 
    # For 4x4x4 block, the output is 4x4. 
    # The full output is 4x9. 
    # The first 4 columns of the output are C[:, 0:4].
    # In W, row index is k*n + i. 
    # For the first block (cols 0-3 of B), the resulting C has cols 0-3.
    # The W matrix for this block should populate rows corresponding to these columns.
    # W444 is 16x49. Its rows correspond to C[i,k] for i,k in 0..3.
    # In the full W (36 rows), the rows for C[:, 0:4] are:
    # k=0: i=0..3 -> rows 0, 1, 2, 3
    # k=1: i=0..3 -> rows 4, 5, 6, 7
    # k=2: i=0..3 -> rows 8, 9, 10, 11
    # k=3: i=0..3 -> rows 12, 13, 14, 15
    # So W[0:16, :] is correct for the first block.
    
    # Block 2: Cols 4-7 of output (uses B cols 4-7)
    U[:, 49:98] = U444
    V[:, 49:98] = V444
    # V rows for B cols 4-7:
    # V index is j*p + k. p=9.
    # B[j, 4] -> row j*9 + 4.
    # V444 is 16x49. Its rows correspond to B[j, k] for j,k in 0..3.
    # We need to map V444 rows (0..15) to V rows (36..51? No, V is 16x114).
    # V is 16 rows (m*p = 4*9=36? No, V is (m*p, rank). m=4, p=9. 36 rows).
    # Wait, V shape is (m*p, rank) = (36, 114).
    # My V444 is 16x49.
    # The rows of V444 correspond to B[j, k] where j=0..3, k=0..3.
    # In the full V, the rows for B[:, 4:8] are:
    # j=0, k=4..7 -> rows 4, 5, 6, 7
    # j=1, k=4..7 -> rows 13, 14, 15, 16
    # j=2, k=4..7 -> rows 22, 23, 24, 25
    # j=3, k=4..7 -> rows 31, 32, 33, 34
    # V444 rows are flattened j*4+k.
    # Row 0 (j=0,k=0) -> V row 4
    # Row 1 (j=0,k=1) -> V row 5
    # ...
    # Row 15 (j=3,k=3) -> V row 34
    
    # We need to permute/insert V444 into V.
    # It's easier to construct V block by block.
    
    # Let's restart the assembly with explicit row mapping.
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: Cols 0-3
    U[:, 0:49] = U444
    # V mapping: V444 row r (j*4+k) -> V row j*9+k
    for r in range(16):
        j, k = divmod(r, 4)
        V[j*9 + k, 0:49] = V444[r, :]
    # W mapping: W444 row r (k*4+i) -> W row k*4+i
    W[0:16, 0:49] = W444
    
    # Block 2: Cols 4-7
    U[:, 49:98] = U444
    for r in range(16):
        j, k = divmod(r, 4)
        # B cols 4-7, so k in 4..7
        V[j*9 + (k+4), 49:98] = V444[r, :]
    # W mapping: W444 row r (k*4+i) -> W row (k+4)*4+i
    for r in range(16):
        k, i = divmod(r, 4)
        W[(k+4)*4 + i, 49:98] = W444[r, :]
        
    # Block 3: Col 8
    U[:, 98:114] = U441
    # V mapping: V441 row r (j*1+0) -> V row j*9+8
    for r in range(16):
        j, k = divmod(r, 1) # k is always 0
        V[j*9 + 8, 98:114] = V441[r, :]
    # W mapping: W441 row r (0*4+i) -> W row 8*4+i
    for r in range(16):
        k, i = divmod(r, 4) # k is always 0
        W[8*4 + i, 98:114] = W441[r, :]
        
    return {"U": U, "V": V, "W": W}
