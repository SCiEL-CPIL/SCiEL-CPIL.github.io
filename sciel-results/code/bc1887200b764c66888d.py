import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 26 decomposition:
    # Block 1: j=0, k=0..4. Optimized 2x1x5 with rank 9 (instead of 10 schoolbook).
    #   For each i in {0,1}, use 5 products with shared A[i,0] factor and
    #   differences in B to cover all k with rank 5 per i.
    # Block 2: j=1,2.
    #   Sub-block 2a: k=0,1 (2x2x2). Strassen, rank 7.
    #   Sub-block 2b: k=2 (2x2x1). Schoolbook, rank 4.
    #   Sub-block 2c: k=3,4 (2x2x2). Strassen, rank 7.
    # Total: 9 + 7 + 4 + 7 = 27... wait, that's 27.
    
    # Let me recount the j=0 optimization.
    # For fixed i, we need: C[i,k] = A[i,0] * B[0,k] for k=0,1,2,3,4.
    # Schoolbook: 5 terms, each A[i,0]*B[0,k].
    # Can we do it in 4? 
    # M1 = A[i,0]*B[0,0]
    # M2 = A[i,0]*B[0,1]
    # M3 = A[i,0]*B[0,2]
    # M4 = A[i,0]*B[0,3]
    # M5 = A[i,0]*B[0,4]
    # That's 5. Can we combine?
    # The tensor for fixed i is: T[j=0, k] = B[0,k] (as a function of k).
    # This is a vector in R^5. The rank of the "matrix multiplication" for 1x1x5 is 5.
    # So for each i, we need rank 5. For 2 i's, rank 10.
    # Wait, but the A factor is A[i,0], which is different for each i.
    # So we can't share across i.
    # Actually, for fixed j=0, the tensor is:
    # C[i,k] = A[i,0] * B[0,k]
    # This is an outer product of vector a (length n=2) and vector b (length p=5).
    # The rank of this 2x5 "matrix" as a bilinear form is... 
    # T[i,0,k] = a_i * b_k.
    # This is a rank-1 tensor in the (i,k) space for fixed j=0? No.
    # The full tensor T[i,j,k] with j=0 only: T[i,0,k] = a_i * b_k.
    # This can be written as sum over r of U[i,0,r] * V[0,k,r] * W[k,i,r].
    # If we set V[0,k,r] = b_k for all r (no, V depends on r).
    # Actually, T[i,0,k] = sum_r U[i,0,r] * V[0,k,r] * W[k,i,r].
    # We want this to equal a_i * b_k for all i,k.
    # If we set W[k,i,r] = 1 for all k,i,r, then we need sum_r U[i,0,r]*V[0,k,r] = a_i*b_k.
    # This is a matrix factorization of the outer product a*b^T, which has rank 1.
    # So we can do it in rank 1!
    # U[i,0,0] = a_i = 1 (coefficient of A[i,0])
    # V[0,k,0] = b_k = 1 (coefficient of B[0,k])
    # W[k,i,0] = 1
    # Then sum_r U[i,0,r]*V[0,k,r]*W[k,i,r] = U[i,0,0]*V[0,k,0]*W[k,i,0] = 1*1*1 = 1.
    # But we need T[i,0,k] = 1 (since it's the coefficient of A[i,0]*B[0,k] in C[i,k]).
    # Yes! T[i,0,k] = 1 for all i,k when j=0.
    # So the entire j=0 block is rank 1?
    # Wait, no. The tensor entry T[i,j,k] is 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # For j=0, the term A[i,0]*B[0,k] appears in C[i,k]. So T[i,0,k] = 1.
    # And T[i,j,k] = 0 for j != 0 in this block.
    # So the sub-tensor for j=0 is: T[i,0,k] = 1 for all i in {0,1}, k in {0,1,2,3,4}.
    # This is a 2x5 matrix of 1's, embedded in the 3rd mode at j=0.
    # As a tensor, it's sum over (i,k) of e_i ⊗ e_0 ⊗ e_k.
    # The rank of this tensor: we can write it as sum_r u_r ⊗ v_r ⊗ w_r.
    # u_r is in R^{n*m}=R^6, v_r is in R^{m*p}=R^{15}, w_r is in R^{p*n}=R^{10}.
    # For the j=0 block, u_r has non-zero only at positions i*3+0, v_r at 0*5+k, w_r at k*2+i.
    # We need sum_r u_r[i*3] * v_r[k] * w_r[k*2+i] = 1 for all i,k.
    # And 0 elsewhere.
    # If we take r=0: u_0[i*3]=1 for i=0,1; v_0[k]=1 for k=0..4; w_0[k*2+i]=1 for all k,i.
    # Then the product is 1*1*1=1. And for other (i',j',k') with j'≠0, u_0[i'*3+j']=0, so product is 0.
    # So rank 1 suffices for the j=0 block!
    
    # Wait, that can't be right. Let me double-check.
    # The decomposition requires: for all i,j,k,
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 (all entries are 1, since every A[i,j]*B[j,k] contributes to C[i,k]).
    
    # Oh! I made an error. T[i,j,k] = 1 for ALL i,j,k, not just j=0.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    # So for j=0, T[i,0,k] = 1 for all i,k. But for j=1, T[i,1,k] = 1 too.
    # So the j=0 block contributes T[i,0,k] = 1, and we need the full sum over all blocks to give 1.
    
    # So if I use rank 1 for j=0, it gives T[i,0,k] += 1. Good.
    # But I also need T[i,1,k] = 1 and T[i,2,k] = 1 from the other blocks.
    
    # So the j=0 block can indeed be rank 1!
    # Let me verify: with one term, U[i*3+0, c0] = 1, V[0*5+k, c0] = 1, W[k*2+i, c0] = 1.
    # Then for (i,0,k): product = 1*1*1 = 1. ✓
    # For (i,j,k) with j≠0: U[i*3+j, c0] = 0 (since j≠0), so product = 0. ✓
    
    # So total rank = 1 (j=0) + 7 (2a) + 4 (2b) + 7 (2c) = 19.
    
    # That's a massive improvement! Let me implement rank 19.
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, rank 1
    # U[i*3+0] = 1 for i=0,1
    # V[0*5+k] = 1 for k=0..4
    # W[k*2+i] = 1 for all k,i
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (Schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

