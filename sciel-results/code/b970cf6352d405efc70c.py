"""Matrix multiplication 3x4x6 with grouped tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: For each pair (i,k), the contribution to C[i,k] is sum_j A[i,j]*B[j,k]
    # This is a dot product of length m=4.
    # We can decompose each dot product using Strassen-like methods for m=4.
    # For m=4, we can use 4-dimensional FFT or Winograd's method.
    # Winograd's method for dot product of length 4 uses 6 multiplications instead of 4.
    # Wait, that's worse. Let me reconsider.
    
    # Actually, for a single dot product of length m, the minimal rank is m (it's a vector).
    # The tensor M(n,m,p) has rank at least n*p = 18 (since each output C[i,k] depends on all A[i,:] and B[:,k]).
    
    # Let's try a rank-18 lower bound approach: can we achieve rank 18?
    # That would require rank = p*n, which means W has full row rank.
    # This is only possible if the decomposition is "minimal" in a specific sense.
    
    # For now, use the correct schoolbook with rank 72
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
# EVOLVE-BLOCK-END
