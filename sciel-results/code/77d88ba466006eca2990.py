"""Matrix multiplication 2x2 by 2x11 tensor decomposition."""
DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    # Try rank 2p = 22
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]  # 4 x 22
    V = [[0] * rank for _ in range(m * p)]  # 22 x 22
    W = [[0] * rank for _ in range(p * n)]  # 22 x 22
    
    # Construction for 2x2 x 2xn:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # Use rank 2n decomposition:
    # For each k, we need 2 terms, but we can share the A coefficients.
    # 
    # Term type 1 (for k=0..p-1): 
    #   U[(0,0), c_k] = 1, V[(0,k), c_k] = 1, W[(k,0), c_k] = 1
    #   This gives A[0,0]*B[0,k] -> C[0,k]
    #
    # Term type 2 (for k=0..p-1):
    #   U[(1,0), c'_k] = 1, V[(0,k), c'_k] = 1, W[(k,1), c'_k] = 1
    #   This gives A[1,0]*B[0,k] -> C[1,k]
    #
    # But we still need A[0,1]*B[1,k] and A[1,1]*B[1,k]
    #
    # This only gives 2p terms covering the B[0,k] parts.
    # We need another 2p for B[1,k] parts. Total 4p = 44.
    
    # That's just schoolbook. Let me think of a better way.
    
    # Alternative: use the fact that we can combine.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Consider: (A[0,0]+A[1,0]) * B[0,k] = C[0,k] + C[1,k] - A[0,1]*B[1,k] - A[1,1]*B[1,k]
    # This doesn't simplify things.
    
    # Let me just return the schoolbook for now and verify it works.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
