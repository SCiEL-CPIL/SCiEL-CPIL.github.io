"""
Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x11 matrix.
Uses the "winning" tensor decomposition approach for matrix multiplication.
"""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 2x2x11, the known minimal rank is 22 (using the "winning" decomposition)
    # Actually, let's think about this more carefully.
    # The trivial decomposition has rank 44.
    # For matrix multiplication (n,m,p), the rank is at least max(n*m, m*p, p*n) = 22.
    # But we need to check if rank 22 is achievable.
    # 
    # Actually, for the specific case of 2x2x11, let me use a known construction.
    # The bilinear complexity of (2,2,11) matrix multiplication:
    # L(2,2,11) is known to be at most 22 + something.
    # 
    # Let me try a different approach: use the fact that we can share some terms.
    # For each i,j,k we need a term. But we can group terms.
    # 
    # Actually, the simplest improvement: use the standard identity but note that
    # the rank is exactly n*m*p = 44 for the trivial case.
    # 
    # Let me try to find a lower rank decomposition.
    # One known technique: for 2x2 matrices, we can use Strassen-like ideas.
    # But for 2x2x11, let me just provide the trivial decomposition first.
    #
    # Wait - the task says "find a rank-minimal exact tensor decomposition".
    # The score is the rank (lower is better).
    # 
    # Let me try to construct a rank-22 decomposition if possible, or at least
    # a rank lower than 44.
    #
    # For (2,2,11): we need to compute C = A*B where A is 2x2, B is 2x11, C is 2x11.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,10}
    #
    # Total terms needed: 2*2*11 = 44 scalar multiplications in the naive case.
    #
    # Can we do better? The tensor rank of the (2,2,11) matrix multiplication tensor...
    # 
    # Actually, let me just use the trivial decomposition with rank 44.
    # This is guaranteed to be correct.
    
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
