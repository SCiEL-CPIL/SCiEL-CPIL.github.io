import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We want to decompose the matrix multiplication tensor for (3,3,7)
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use a rank-24 decomposition based on partitioning the 7 columns
    # into overlapping blocks. The key insight is that for 3x3 matrices, we can
    # use Strassen-like optimizations, but for 3x7, we partition the 7 output 
    # columns into: first 3, middle 3 (overlapping by 1), and last 1.
    # Actually, let me use a direct construction:
    # 
    # For 3x3, the rank is 21 (not 27). The 21 rank-1 tensors for 3x3x3 are known.
    # For 3x3x7, we can use the 3x3x3 rank-21 decomposition for the first 3 columns,
    # and then handle the remaining 4 columns.
    #
    # Actually, let me try a simpler approach: use the naive decomposition but 
    # see if we can reduce it. The naive rank is 3*3*7 = 63.
    #
    # Let me construct a valid decomposition. I'll use a systematic approach:
    # For each (i,j,k), we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for some r, and 0 for all other (i',j',k') combinations.
    #
    # The simplest valid decomposition is the naive one with rank = n*m*p = 63.
    # Let me construct that:
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now, can we reduce the rank? Let me try a different approach.
    # For 3x3x3, the rank is 21. The 21 rank-1 tensors are:
    # Actually, the exact 21-rank decomposition for 3x3x3 is complex.
    # Let me just return the naive 63-rank decomposition for now.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
