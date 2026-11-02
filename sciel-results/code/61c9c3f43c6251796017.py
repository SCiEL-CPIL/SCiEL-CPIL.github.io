"""Matrix multiplication 2,2,13 - rank-26 decomposition via column-additive structure"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: exploit the fact that for a 2x2 A and a single column b of B (2x1),
    # A*b has rank at most 2, and we can decompose using the structure of A.
    # 
    # For each column k of B, we need to compute:
    #   C[:,k] = A @ B[:,k]
    # where A is 2x2 and B[:,k] is 2x1.
    #
    # The tensor for multiplying a 2x2 matrix by a 2x1 vector has rank 4 (schoolbook).
    # However, we can use a more clever decomposition.
    #
    # Actually, let's think differently. The full problem is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is a bilinear map. The tensor T_{i,k,j} = delta_{i,j} (in some sense).
    # 
    # A standard approach: for each column k, the map B[:,k] -> C[:,k] is linear,
    # given by multiplication by A. But A varies, so it's bilinear.
    #
    # Let's use the identity: for any 2x2 matrix A and 2x1 vector v,
    #   A*v = (v[0]*A[:,0] + v[1]*A[:,1])
    # This doesn't directly help with rank reduction.
    #
    # Alternative: use the fact that the multiplication tensor for (2,2,1) has rank 4,
    # and for (2,2,13) we have 13 independent columns. If we could share factors across
    # columns, we might do better. But the standard Kronecker structure gives rank 4*13=52
    # for schoolbook, or with Strassen blocks... 
    #
    # Wait - the current program uses rank 46. Can we do rank 26?
    # 
    # The directive mentions rank n*p = 26. Let me think about this.
    # 
    # For a 2x2 matrix A and a 2x13 matrix B, the product C = A@B is 2x13.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Consider the decomposition where each term is:
    #   U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    # 
    # For rank 26 = n*p, we can use one term per (i,k) pair... but that's 2*13=26.
    # For term (i,k), we'd need:
    #   sum_r U_{(i,j),r} V_{(j,k),r} = A[i,j] B[j,k] for all j
    # 
    # This means for fixed (i,k), we need a decomposition of the outer product 
    # (A[i,:])^T (B[:,k]) into a sum of rank-1 matrices in the j-index.
    # Since A[i,:] and B[:,k] are both length-2 vectors, their outer product is 2x2
    # and can be decomposed into at most 2 rank-1 terms (actually rank 1 if they're
    # nonzero). But we need this to work uniformly for all A and B.
    #
    # Actually, the bilinear identity approach: 
    #   A @ B = sum_r (u_r * A) * (v_r * B^T) * w_r ... 
    #
    # Let me reconsider. The rank-26 approach:
    # For each pair (i,k) where i in {0,1} and k in {0,...,12}, we create one term.
    # The term should contribute to C[i,k] only.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # If we set for term r corresponding to (i,k):
    #   U[r] has 1s at positions (i,0) and (i,1) -> but that's two entries
    #   V[r] has 1s at positions (0,k) and (1,k)
    #   W[r] has 1 at position (k,i)
    # 
    # But that would give sum_j U[r,j] V[r,j] = V[r,0]+V[r,1] which is B[0,k]+B[1,k],
    # not what we want.
    #
    # The correct rank-26 construction:
    # For each (i,k), use TWO terms (one for j=0, one for j=1):
    # That gives 2*2*13 = 52 terms (schoolbook).
    #
    # To get rank 26, we need to be clever. Let's use:
    # For each k (column of B), and for the 2x2 -> 2x1 multiplication:
    # The rank of the (2,2,1) multiplication tensor is 4 (schoolbook) or can it be lower?
    # No, it's 4 for 2x2x1.
    # 
    # So 13 columns * 4 = 52 is the schoolbook rank.
    # Strassen doesn't help here because Strassen requires 2x2x2.
    # 
    # But wait - maybe we can use a different grouping. What if we pair columns?
    # The current approach uses Strassen for pairs of columns, getting 7 per 2 columns.
    # 6 pairs * 7 + 4 for the last = 42 + 4 = 46. That's what we have.
    #
    # Can we do better than 46? The theoretical lower bound for (2,2,13) is...
    # The rank of (n,m,p) multiplication is at least max(n*m, m*p, p*n) / something.
    # 
    # Actually, let me try a different decomposition. What if we use the following:
    # For the (2,2,13) problem, we can view it as:
    #   C = A @ B where A is 2x2, B is 2x13
    # 
    # We can write A = [a0; a1] where a0, a1 are 1x2 row vectors.
    # C[i,k] = a_i . B[:,k]
    # 
    # Now, B[:,k] is a 2x1 vector. Let's define for each k:
    #   s_k = B[0,k] + B[1,k]
    #   d_k = B[0,k] - B[1,k]
    # 
    # Then B[0,k] = (s_k + d_k)/2, B[1,k] = (s_k - d_k)/2
    # 
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #        = A[0,0]*(s_k+d_k)/2 + A[0,1]*(s_k-d_k)/2
    #        = (A[0,0]+A[0,1])*s_k/2 + (A[0,0]-A[0,1])*d_k/2
    # 
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #        = (A[1,0]+A[1,1])*s_k/2 + (A[1,0]-A[1,1])*d_k/2
    # 
    # So for each k, we need:
    #   Term 1: (A[0,0]+A[0,1])*s_k -> contributes to C[0,k] with factor 1/2
    #   Term 2: (A[0,0]-A[0,1])*d_k -> contributes to C[0,k] with factor 1/2
    #   Term 3: (A[1,0]+A[1,1])*s_k -> contributes to C[1,k] with factor 1/2
    #   Term 4: (A[1,0]-A[1,1])*d_k -> contributes to C[1,k] with factor 1/2
    # 
    # That's still 4 terms per column, total 52. No improvement.
    #
    # Let me try yet another approach. What if we combine across columns?
    # Define S = sum_k B[:,k] (a 2x1 vector) and D_k = B[:,k] for each k.
    # This doesn't seem to help.
    #
    # I think 46 is close to optimal for this approach. Let me just make sure
    # the current implementation is correct and return it.
    
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        add_term(
            {0: 1, 3: 1},
            {k0: 1, 13 + k1: 1},
            {2 * k0: 1, 2 * k1 + 1: 1}
        )
        add_term(
            {2: 1, 3: 1},
            {k0: 1},
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, 13 + k1: -1},
            {2 * k1: 1, 2 * k1 + 1: 1}
        )
        add_term(
            {3: 1},
            {13 + k0: 1, k0: -1},
            {2 * k0: 1, 2 * k0 + 1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {13 + k1: 1},
            {2 * k0: -1, 2 * k1: 1}
        )
        add_term(
            {2: 1, 0: -1},
            {k0: 1, k1: 1},
            {2 * k1 + 1: 1}
        )
        add_term(
            {1: 1, 3: -1},
            {13 + k0: 1, 13 + k1: 1},
            {2 * k0: 1}
        )
    
    k = 12
    add_term({0: 1}, {k: 1}, {2 * k: 1})
    add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
    add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
