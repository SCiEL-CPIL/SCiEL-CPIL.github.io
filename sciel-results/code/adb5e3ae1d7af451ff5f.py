"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We'll use Strassen's algorithm adapted for this case.
    # Split A (3x3) into blocks: A_top = A[0:2, :] (2x3), A_bottom = A[2, :] (1x3)
    # Split B (3x7) columns into pairs: for k = 0,1,2,3,4,5: pair (k, k+1), and k=6 single.
    
    # For each pair of columns (k, k+1):
    #   Compute A_top * B[:, k:k+2] (2x3 * 3x2 = 2x2) using rank-5 Strassen-like
    #   Compute A_bottom * B[:, k:k+2] (1x3 * 3x2 = 1x2) using rank-3 schoolbook
    # For the single column k=6:
    #   Compute A * B[:, 6] (3x3 * 3x1 = 3x1) using rank-9 schoolbook... 
    #   or rank-3 for each row, so 3 total.
    
    # Let me define the Strassen-like decomposition for 2x3 * 3x2 -> 2x2.
    # The 7 products of standard Strassen for 2x2*2x2 don't directly apply
    # because we have 3 columns in A and 3 rows in B.
    
    # Alternative: use 6 products for 2x3 * 3x2:
    # This is equivalent to computing a 2x2 matrix from outer products.
    # Minimum rank of this tensor is 4 (since it's like 2 separate 2x1 * 1x2... no).
    
    # Actually, 2x3 * 3x2 can be done in 4 multiplications:
    # Let a = A[0,0]+A[0,1], b = A[0,2], c = A[1,0]+A[1,1], d = A[1,2]
    # Let e = B[0,k]+B[1,k], f = B[2,k], g = B[0,k+1]+B[1,k+1], h = B[2,k+1]
    # M1 = a * e, M2 = b * f, M3 = c * g, M4 = d * h
    # C[0,k] = M1 - (A[0,1]*B[1,k]) + M2 ... no this doesn't work cleanly.
    
    # Let me just use schoolbook for everything but with a smarter grouping.
    # 
    # Actually, the simplest improvement: note that for 3x3 * 3x7,
    # we can view it as 3x3 * 3x3 + 3x3 * 3x3 + 3x3 * 3x1.
    # If we use Strassen (rank 7) for each 3x3*3x3 block and rank 3 for 3x3*3x1,
    # total rank = 7 + 7 + 3 = 17.
    
    # But wait - the Strassen decomposition for 3x3*3x3 gives us
    # U (9x7), V (9x7), W (9x7) for the standard 3x3*3x3 tensor.
    # For 3x3*3x1, we need U (9x3), V (3x3), W (3x3).
    
    # The issue: the column indices in V and W are different.
    # For the first 3x3*3x3 block (cols 0-2 of B), V has entries at col indices 0,1,2.
    # For the second 3x3*3x3 block (cols 3-5 of B), V has entries at col indices 3,4,5.
    # For the 3x3*3x1 block (col 6 of B), V has entries at col index 6.
    
    # So we can concatenate! Let's build it.
    
    # First, define Strassen for 3x3 * 3x3:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # B = [[B00, B01, B02],
    #      [B10, B11, B12],
    #      [B20, B21, B22]]
    # 
    # Standard Strassen for 3x3 uses 7 products? No, standard Strassen is for 2x2.
    # For 3x3, the best known is rank 23 (or maybe lower). Strassen's algorithm
    # recursively applied to 3x3 doesn't give a simple formula.
    
    # I'll use schoolbook for the 3x3*3x3 blocks too, giving rank 9 each.
    # Total: 9 + 9 + 3 = 21. Still better than 63!
    
    # Actually wait, that's not how it works. The full 3x3*3x7 tensor is NOT
    # a direct sum of smaller tensors in the way I'm thinking. Each entry of C
    # depends on all of A and the corresponding column of B.
    
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # This is a bilinear map. The tensor is T[i,j,k] = 1 for all i,j,k.
    # We want to write T = sum_c u_c ⊗ v_c ⊗ w_c where:
    #   u_c[i,j] contributes to U[i*m+j, c]
    #   v_c[j,k] contributes to V[j*p+k, c]  
    #   w_c[k,i] contributes to W[k*n+i, c]
    # and sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    # Schoolbook: T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}, rank = 27*... 
    # No, rank = n*m*p = 3*3*7 = 63.
    
    # Let me try a decomposition with rank 21:
    # Group by output column k. For each k, we have a 3x3 sub-tensor (i,j) -> value.
    # For fixed k, T_k[i,j] = 1 for all i,j. This is a 3x3 all-ones matrix.
    # We can write: for each k, T_k = sum_j e_j ⊗ e_{j,k} ⊗ ... 
    # 
    # Actually, for fixed k: sum_j A[i,j] * B[j,k].
    # This can be done in 3 multiplications (one per j).
    # So for 7 columns, we need 7*3 = 21 multiplications.
    
    # Decomposition: for each j in {0,1,2} and k in {0,...,6}:
    #   U[i*m+j, c] = 1 for all i  (where c = j*7 + k)
    #   V[j*p+k, c] = 1
    #   W[k*n+i, c] = 1 for all i
    
    # Wait, that's not right. Let me be more careful.
    
    # For term c corresponding to (j, k):
    #   U[i, j] = 1 for all i? No.
    #   We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # If for term c=(j0,k0): U[i,j0,c]=1 for all i, V[j0,k0,c]=1, W[k0,i,c]=1 for all i,
    # then this term contributes 1 to (i,j0,k0) for all i, and 0 elsewhere.
    # That's not what we want.
    
    # The correct schoolbook: for each (i,j,k): term has U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # This gives exactly 1 at (i,j,k) and 0 elsewhere. Rank = 63.
    
    # Can we do better? For fixed k, we need: sum_j A[i,j]*B[j,k] for all i.
    # This is a 3x3 matrix product (A times column k of B).
    # Minimum rank for 3x3 matrix-vector product is 3 (you need at least 3 mults).
    # So for 7 columns, minimum rank >= 21.
    
    # Can we achieve rank 21? Let's try:
    # For each j in {0,1,2} and k in {0,...,6}, define term c(j,k):
    #   U[i*m + j, c] = 1 for all i in {0,1,2}
    #   V[j*p + k, c] = 1
    #   W[k*n + i, c] = 1 for all i in {0,1,2}
    
    # Check: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # = sum over all (j0,k0) of U[i,j0,c(j0,k0)] * V[j0,k0,c(j0,k0)] * W[k0,i,c(j0,k0)]
    # Only term where j0=j and k0=k contributes (since V[j0,k0,c] is nonzero only when j0=j, k0=k)
    # Wait no, V[j*p+k, c] = 1 means V is nonzero at position (j,k) for this term.
    # But in the sum, we have V[j,k,c] where j,k are the free indices.
    
    # Let me re-index. The sum is over c:
    #   (sum_c U[i*m+j, c]) ... no, it's:
    #   C[i,k] = sum_{j,c} A[i,j] * V[j*p+k, c] * W[k*n+i, c] * ... 
    #   
    # I think the convention is:
    #   C[i,k] = sum_{j, c} A[i,j] * B[j,k] where the decomposition gives
    #   A[i,j]*B[j,k] -> sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # For my proposed term c=(j0,k0):
    #   U[i*m+j0, c] = 1 for all i
    #   V[j0*p+k0, c] = 1
    #   W[k0*n+i, c] = 1 for all i
    
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]:
    # Only terms where j0=j and k0=k contribute to V being nonzero at (j,k).
    # Wait, V[j*p+k, c] is the entry of V at row j*p+k, column c.
    # For term c=(j0,k0), V[j0*p+k0, c] = 1, and V[other, c] = 0.
    # So V[j*p+k, c] = 1 if and only if j=j0 and k=k0.
    
    # Similarly, U[i*m+j, c] = 1 if j=j0 (for any i), and W[k*n+i, c] = 1 if k=k0 (for any i).
    
    # So the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if and only if j=j0, k=k0.
    # And when j=j0, k=k0: the value is 1 (since U and W are 1 for all i).
    
    # Sum over c: only c=(j,k) contributes, giving value 1. 
    # So sum_c ... = 1 for all i,j,k. 
    
    # This works! Rank = 3*7 = 21.
    
    rank = m * p  # 3 * 7 = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
            V[j * p + k][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
