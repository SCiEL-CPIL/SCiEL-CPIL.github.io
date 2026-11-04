"""Tensor decomposition for 2x4 by 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    
    # Build the multiplication tensor T[i,j,k] where (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # We want to find U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    # Note: W uses trace convention W[k*n+i, r]
    
    # For 2x4 by 4x6, we can use a block decomposition approach.
    # Split the 4 columns of A into two 2-column blocks, and 4 rows of B into two 2-row blocks.
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x2
    # C = A*B = A1*B1 + A2*B2
    
    # For each 2x2 by 2x2 multiplication, we can use rank 7 (Strassen) or rank 8 (schoolbook).
    # With Strassen rank 7 for each 2x2x2 block: total rank = 7*2 = 14
    # But we need to be careful about the indexing.
    
    # Let me use the schoolbook approach first to ensure correctness, then try to optimize.
    # Schoolbook: rank = 2*4*6 = 48
    
    # Actually, let me think about this more carefully.
    # The tensor has dimensions n x m x p = 2 x 4 x 6
    # We want a decomposition into sum of rank-1 tensors.
    
    # Let's try to exploit structure. 
    # A is 2x4, B is 4x6.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # We can group j=0,1 and j=2,3:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # This is just 4 terms per (i,k) pair, so naive rank is 2*4*6/... 
    # Actually each term A[i,j]*B[j,k] contributes to all k for fixed i,j.
    
    # Let me just implement the schoolbook decomposition with rank 48.
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
