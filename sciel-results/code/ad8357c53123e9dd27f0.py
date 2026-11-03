"""Matrix multiplication 2x3 by 3x10 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # We will construct a decomposition of rank 55:
    # - Part 1: 2x2 x 2x10 multiplication using block Strassen (5 blocks of 2x2, rank 7 each = 35)
    # - Part 2: 2x1 x 1x10 multiplication (outer product, rank 20)
    total_rank = 55
    U = np.zeros((n * m, total_rank))  # (6, 55)
    V = np.zeros((m * p, total_rank))  # (30, 55)
    W = np.zeros((p * n, total_rank))  # (20, 55)
    
    # Part 1: 2x2 x 2x10
    # A is 2x2 (rows 0,1; cols 0,1), B is 2x10 (rows 0,1; cols 0..9)
    # Split B into 5 blocks of 2x2: B[:, 0:2], B[:, 2:4], ..., B[:, 8:10]
    # For each 2x2 block, use Strassen's algorithm (rank 7)
    
    # Strassen's algorithm for 2x2 x 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We'll define the Strassen decomposition for a single 2x2 block
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (row-major)
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (row-major)
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 (row-major)
    
    # For each of the 5 blocks, we add 7 rank-1 terms
    for block in range(5):
        base = block * 7
        # B columns for this block: 2*block, 2*block+1
        b_col0 = 2 * block
        b_col1 = 2 * block + 1
        
        # M1 = (A00+A11)*(B00+B11)
        # A part: A00=1, A11=1 -> U[0]=1, U[3]=1
        # B part: B00=1, B11=1 -> V[0]=1, V[3]=1
        # C part: C00=1, C11=1 -> W[0]=1, W[3]=1
        U[0, base] = 1
        U[3, base] = 1
        V[0, base] = 1
        V[3, base] = 1
        W[0, base] = 1
        W[3, base] = 1
        
        # M2 = (A10+A11)*B00
        # A part: A10=1, A11=1 -> U[2]=1, U[3]=1
        # B part: B00=1 -> V[0]=1
        # C part: C10=1, C11=1 -> W[2]=1, W[3]=1
        U[2, base + 1] = 1
        U[3, base + 1] = 1
        V[0, base + 1] = 1
        W[2, base + 1] = 1
        W[3, base + 1] = 1
        
        # M3 = A00*(B01-B11)
        # A part: A00=1 -> U[0]=1
        # B part: B01=1, B11=-1 -> V[1]=1, V[3]=-1
        # C part: C01=1, C11=1 -> W[1]=1, W[3]=1
        U[0, base + 2] = 1
        V[b_col1, base + 2] = 1
        V[3, base + 2] = -1
        W[1, base + 2] = 1
        W[3, base + 2] = 1
        
        # M4 = A11*(B10-B00)
        # A part: A11=1 -> U[3]=1
        # B part: B10=1, B00=-1 -> V[2]=1, V[0]=-1
        # C part: C00=1, C10=1 -> W[0]=1, W[2]=1
        U[3, base + 3] = 1
        V[2, base + 3] = 1
        V[0, base + 3] = -1
        W[0, base + 3] = 1
        W[2, base + 3] = 1
        
        # M5 = (A00+A01)*B11
        # A part: A00=1, A01=1 -> U[0]=1, U[1]=1
        # B part: B11=1 -> V[3]=1
        # C part: C00=-1, C01=1 -> W[0]=-1, W[1]=1
        U[0, base + 4] = 1
        U[1, base + 4] = 1
        V[3, base + 4] = 1
        W[0, base + 4] = -1
        W[1, base + 4] = 1
        
        # M6 = (A10-A00)*(B00+B10)
        # A part: A10=1, A00=-1 -> U[2]=1, U[0]=-1
        # B part: B00=1, B10=1 -> V[0]=1, V[2]=1
        # C part: C11=1 -> W[3]=1
        U[2, base + 5] = 1
        U[0, base + 5] = -1
        V[0, base + 5] = 1
        V[2, base + 5] = 1
        W[3, base + 5] = 1
        
        # M7 = (A01-A11)*(B01+B11)
        # A part: A01=1, A11=-1 -> U[1]=1, U[3]=-1
        # B part: B01=1, B11=1 -> V[1]=1, V[3]=1
        # C part: C00=1 -> W[0]=1
        U[1, base + 6] = 1
        U[3, base + 6] = -1
        V[b_col1, base + 6] = 1
        V[3, base + 6] = 1
        W[0, base + 6] = 1
    
    # Part 2: 2x1 x 1x10
    # A is 2x1 (rows 0,1; col 2), B is 1x10 (row 2; cols 0..9)
    # C = A * B, so C[i,k] = A[i,2] * B[2,k]
    # This is an outer product of A[:,2] and B[2,:]
    # Rank = 2 * 10 = 20
    # For each (i,k) pair, we have a rank-1 term
    base2 = 35
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0..9
            col = base2 + i * p + k
            U[i * m + 2, col] = 1  # A[i,2]
            V[2 * p + k, col] = 1  # B[2,k]
            W[k * n + i, col] = 1  # C[i,k]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
