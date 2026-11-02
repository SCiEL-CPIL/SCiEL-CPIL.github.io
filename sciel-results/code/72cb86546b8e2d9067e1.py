"""Matrix multiplication 3x3 * 3x4 with reduced rank decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4  # A is n x m, B is m x p, C is n x p
    
    # Rank-15 decomposition for 3x3 * 3x4
    # Strategy: For each output column k (4 cols) and each "group" of middle indices
    # We use the fact that sum_j A[i,j]*B[j,k] can be decomposed
    # 
    # Construction: rank = m*p + (n-1)*p = 3*4 + 2*4 = 20? No.
    # 
    # Better: Use the "Laderman-like" approach adapted.
    # For 3x3 * 3x4, we can achieve rank 15.
    #
    # Direct construction:
    # For each j in range(m), for each k in range(p):
    #   term: u_i = A[i,j], v_j = B[j,k], w_k = C[i,k]
    # That's m*p = 12 terms, but each term is rank-1 in the tensor sense
    # only if we fix i. We need i to vary.
    #
    # Correct rank-1: u_i * v_j * w_k where u depends on i, v on j, w on k.
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    #
    # For T[i,j,k] = 1 (all ones tensor for valid indices), we need:
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    #
    # This is equivalent to: the 3x3x4 all-ones tensor has rank...
    # Actually no! The matrix multiplication tensor is NOT all ones.
    # T[i,j,k] = 1 means: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    # So yes, T is the all-ones tensor of shape (3,3,4) in the appropriate indexing.
    #
    # Wait, that's not right either. Let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] is the coefficient for output C[i,k]
    # 
    # The decomposition: C[i,k] = sum_{i',j'} A[i',j'] * B[j',k'] * T[i',j',k',i,k]
    # Actually: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # So the tensor T[i,j,k] (where i indexes A row, j indexes A col/B row, k indexes B col)
    # has T[i,j,k] = 1 if this term contributes to C[i,k], which it always does.
    # 
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of n*m*p = 36 equations. We want minimal rank r.
    #
    # For the all-ones tensor of shape (n*m, m*p, p*n) with the constraint that
    # only entries where the indices are "compatible" (i.e., the j in U matches j in V,
    # and k in V matches k in W, and i in W matches i in U) need to be 1, and all
    # other entries can be anything (they don't appear in the product).
    #
    # Actually, the constraint is only on the "compatible" triples. Let me define:
    # For each (i,j,k), the entry U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    #
    # Key insight: This is equivalent to finding a low-rank factorization of a specific
    # structured matrix. Let me think of it as: for each r, we have vectors
    # u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} such that for all compatible (i,j,k):
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # A simple construction: use rank = m*p = 12? No, that's not enough.
    #
    # Let me try rank = 15 with a specific construction.
    # 
    # Alternative: use the "middle" decomposition. For each j, define:
    #   a_{i,j} = A[i,j], b_{j,k} = B[j,k]
    #   C[i,k] = sum_j a_{i,j} b_{j,k}
    #
    # I can write this as: C = A * B, and use a rank decomposition of the bilinear map.
    #
    # Known result: rank of matrix multiplication for (n,m,p) is at most n*m*p,
    # and for small cases, specific values are known.
    #
    # For 3x3 * 3x4, let me just use a constructive approach with rank 15.
    #
    # Construction idea: 
    # - 3 terms for the "diagonal" in j: for each j, u[i*m+j]=delta[i,j], v[j*p+k]=delta[j,k]... 
    #   No, that doesn't work directly.
    #
    # Let me use a different approach: the tensor is "separable" in a structured way.
    # For fixed j, the sub-tensor over (i,k) is all ones (3x4).
    # So T = sum_j (e_j ⊗ e_j ⊗ J_{3,4}) where J is the all-ones matrix... no.
    #
    # Actually: T[i,j,k] = 1 for all i,j,k. This is the tensor 1_{3} ⊗ 1_{3} ⊗ 1_{4}.
    # The rank of a Kronecker product of vectors is 1! So the all-ones tensor has rank 1??
    # No, that's wrong. The all-ones tensor of shape (3,3,4) can be written as
    # u ⊗ v ⊗ w where u_i = 1, v_j = 1, w_k = 1. So yes, it has rank 1!
    #
    # But wait, that can't be right for matrix multiplication. Let me re-examine.
    #
    # Oh! I see my error. The matrix multiplication tensor is NOT the all-ones tensor
    # in the (i,j,k) space. Let me re-read the problem.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix 
    #  by a 3 by 4 matrix."
    #
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # A rank-1 term in this bilinear map is: (A, B) -> (sum_{i,j} U[i*m+j, r] A[i,j]) * 
    #   (sum_{j,k} V[j*p+k, r] B[j,k]) * ... no, that's not how it works.
    #
    # Actually, a rank-1 term for the matrix multiplication tensor is:
    # C[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # summed over i,j,k,r.
    #
    # So the condition is: for all i,j,k,
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # And this must hold ONLY for the "compatible" (i,j,k) triples, which is all of them
    # since every (i,j,k) triple is compatible (i from 0..2, j from 0..2, k from 0..3).
    #
    # So yes, T[i,j,k] = 1 for all 36 triples. And this tensor IS the outer product
    # of three all-ones vectors, so it has rank 1!
    #
    # But that would mean we can do matrix multiplication with rank 1, which is absurd.
    # Let me re-examine what "rank" means here.
    #
    # I think I'm confusing the tensor with the bilinear map. Let me look at this more carefully.
    #
    # The matrix multiplication tensor T is in R^{(n*m) x (m*p) x (p*n)} where
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j', k=k', i=i', else 0.
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    #
    # THIS is not the all-ones tensor. It's a structured tensor where only "compatible" 
    # triples have value 1.
    #
    # So the condition is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c.
    # Where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
    #
    # Now this is a non-trivial tensor. Let me think about its rank.
    #
    # For the schoolbook: rank = n*m*p = 36, with U[i*m+j, i*m+j] = 1, etc.
    #
    # Can we do better? The tensor has a lot of structure. Let me try to find a lower rank.
    #
    # Key observation: T[a,b,c] depends on the "overlap" of indices. Specifically,
    # T[i*m+j, j*p+k, k*n+i] = 1. This means the middle index j appears in both a and b,
    # the output column k appears in both b and c, and the input row i appears in both a and c.
    #
    # This is a "cyclic" structure. The rank of such tensors is related to the 
    # "slice rank" or "border rank" of matrix multiplication.
    #
    # For 3x3 * 3x4, I'll attempt a rank-15 construction based on known results for
    # small matrix multiplication.
    #
    # Actually, let me just try a computational approach: set up the equations and solve.
    # But that might be slow. Let me try a constructive approach.
    #
    # Construction for rank r:
    # For each r in range(r), define U[:,r], V[:,r], W[:,r].
    # We need: for all i,j,k: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    # And: for all incompatible (a,b,c): sum_{r'} U[a, r'] * V[b, r'] * W[c, r'] = 0
    #
    # The "incompatible" conditions are crucial and make this hard.
    #
    # Let me try rank = 15 with a specific pattern. I'll use the following idea:
    # Group the 36 compatible triples into 15 groups, each handled by one rank-1 term.
    #
    # Actually, for this specific case, let me just use the schoolbook (rank 36) 
    # and see if I can prune. Or better, let me use a known low-rank construction.
    #
    # For now, let me return the schoolbook decomposition which is guaranteed correct.
    # The rank is 36.
    
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
