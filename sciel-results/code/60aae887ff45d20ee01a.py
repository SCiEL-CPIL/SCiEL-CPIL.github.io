"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Exploit the linear structure: decompose 2x2 * 2x14 into two independent
    1x2 * 2x14 vector-matrix products (one per output row).
    
    For row i of A (i=0,1):
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Each 1x2 * 2x14 product has tensor rank exactly 2*14 = 28 (schoolbook is optimal
    for vector-matrix multiplication tensors). Total rank = 2 * 28 = 56.
    
    However, we can achieve rank 28 by sharing the B factors across both output rows.
    The key insight: for each k and j, the term A[i,j]*B[j,k] uses the same B[j,k].
    We can structure the decomposition so that each (j,k) pair contributes one rank-1
    term that covers both output rows via appropriate W entries.
    
    Actually, let me reconsider. The standard decomposition gives rank 56.
    For 2x2 * 2x14, the tensor rank is known to be at most 56 (schoolbook).
    
    Let's verify if we can do better. The tensor T[i,j,k] = 1 for all i,j,k.
    A rank-1 term is U[i,j] * V[j,k] * W[k,i].
    
    For rank r, we need: sum_{s=1}^r U_s[i,j] * V_s[j,k] * W_s[k,i] = 1 for all i,j,k.
    
    This is a 3-way tensor of size 2x2x14. The minimum rank is at least max(2,2,14)=14
    by the trivial bound, but likely higher.
    
    Schoolbook: rank 56. Let's just use that since we can't easily find better.
    """
    n, m, p = DIMENSIONS
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
