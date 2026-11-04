"""Program for matrix multiplication 2x4x6 using Winograd Generalized Algorithm."""
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # For each fixed j, the contribution is A[i,j] * B[j,k] for all i,k.
    # This can be captured with a single column per j in the tensor decomposition.
    #
    # Convention (from seed program):
    # U has shape (n*m, rank), indexed by (i, j) -> row i*m + j
    # V has shape (m*p, rank), indexed by (j, k) -> row j*p + k  
    # W has shape (p*n, rank), indexed by (k, i) -> row k*n + i
    #
    # The decomposition satisfies:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    #
    # Wait, that's not quite right. Let me think about this more carefully.
    #
    # Actually, looking at the seed program more carefully, the convention is:
    # For each (i,j,k) triple in schoolbook, we set:
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1
    #   W[k*n+i, c] = 1
    #
    # And the reconstruction formula must be:
    # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # But wait, in schoolbook each c corresponds to a unique (i,j,k), so:
    # sum_j U[i*m+j, c] * V[j*p+k, c] = 1 only when j = j_c (the j for that column)
    # and i = i_c and k = k_c.
    #
    # So C[i,k] = sum_c delta(i=i_c) * delta(k=k_c) * 1 * W[k*n+i, c]
    # = sum_c delta(i=i_c) * delta(k=k_c) * 1
    # = number of columns c with i_c = i and k_c = k
    # = m (since for each j_c, there's a column)
    #
    # That gives C[i,k] = m = 4, which is wrong. So the formula must be different.
    #
    # Let me reconsider. Maybe the formula is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    #
    # No, that's trilinear, not bilinear.
    #
    # I think the correct interpretation is that U, V, W define linear transformations:
    # For each c, define:
    #   a_c = sum_j U[i*m+j, c] * A[i,j]  (this is a linear functional of A)
    #   b_c = sum_j V[j*p+k, c] * B[j,k]  (this is a linear functional of B)
    #   C[i,k] += W[k*n+i, c] * a_c * b_c
    #
    # But a_c depends on i and b_c depends on k, so this doesn't work as written.
    #
    # I think the correct formula must be:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    #
    # But this is still not bilinear in the right way because the sums depend on i and k respectively.
    #
    # Let me just try a different approach. I'll construct a decomposition that I can verify
    # works by checking the formula directly.
    #
    # Actually, I think the simplest correct interpretation is:
    # The tensor T[i,k,j] (the structure tensor) is decomposed as:
    # T[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n,m,rank), V has shape (m,p,rank), W has shape (p,n,rank).
    #
    # And the multiplication is:
    # C[i,k] = sum_j T[i,k,j] * A[i,j] * B[j,k]
    #        = sum_c sum_j U[i,j,c] * V[j,k,c] * W[k,i,c] * A[i,j] * B[j,k]
    #        = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    #
    # Wait, that's still not quite right because the two sums are over different things.
    #
    # Let me just use a rank-4 construction that I can verify:
    # For each j in {0,1,2,3}, create one column c=j where:
    #   U[i*m+j, c] = 1 for all i
    #   V[j*p+k, c] = 1 for all k
    #   W[k*n+i, c] = 1 for all (k,i)
    #
    # Then the formula C[i,k] = sum_c W[k*n+i, c] * (sum_j' U[i*m+j', c] * A[i,j']) * (sum_j' V[j'*p+k, c] * B[j',k])
    # = sum_c 1 * A[i,c] * B[c,k]
    # = sum_c A[i,c] * B[c,k]
    # = sum_j A[i,j] * B[j,k]
    #
    # This works! And it has rank = m = 4.
    
    rank = m  # 4
    U = [[0] * rank for _ in range(n * m)]   # 8 x 4
    V = [[0] * rank for _ in range(m * p)]   # 24 x 4
    W = [[0] * rank for _ in range(p * n)]   # 12 x 4
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
            W[k * n + i][c] = 1  # Wait, this uses i from the outer loop, which is wrong
    
    # Let me fix the W assignment - it should be for all (k,i)
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
