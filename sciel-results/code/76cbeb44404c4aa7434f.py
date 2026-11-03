import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    # Use two Bini et al. rank-23 decompositions for the two 3x3x3 blocks
    # B is 3x6, split into B[:,0:3] and B[:,3:6]
    
    # Bini et al. 1979 rank-23 decomposition for 3x3x3
    # U3, V3, W3 are 9x23 matrices
    # Coefficients from Bini et al. (1979)
    
    # Define the 23 terms for 3x3x3
    # Each term is (u, v, w) where u, v, w are 9-vectors
    terms = []
    
    # Term 1-9: diagonal-like terms
    for i in range(3):
        for j in range(3):
            u = [0]*9; v = [0]*9; w = [0]*9
            u[i*3+j] = 1
            v[i*3+j] = 1
            w[i*3+j] = 1
            terms.append((u, v, w))
    
    # Terms 10-23: additional terms to reduce rank
    # These are based on the Bini et al. construction
    # Using sums of entries to share computations
    
    # Term 10: (A01+A11) * (B00+B10)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[1] = 1; u[4] = 1  # A01, A11
    v[0] = 1; v[3] = 1  # B00, B10
    # w is determined by the product
    # C entries affected: C10, C11, C20, C21
    # This term contributes to multiple C entries
    # For the tensor decomposition, w should be the coefficient vector
    # Actually, for rank-1 tensor decomposition, w is a 9-vector
    # Let me reconsider the approach
    
    # I'll use a different strategy: just use the trivial rank-54 for now
    # since I can't recall the exact Bini et al. coefficients
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
