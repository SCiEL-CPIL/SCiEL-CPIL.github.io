import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 2, 3, 9

    # Structural insight: C = A*B where A is 2x3, B is 3x9
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Index: i in 0..1 (row of C), k in 0..2 (inner), j in 0..8 (col of C)
    # U index: i*m + k = i*3 + k (6 entries)
    # V index: k*p + j = k*9 + j (27 entries)  
    # W index: j*n + i = j*2 + i (18 entries)
    #
    # Factorization: think of A as 2x3 = (2x1) * (1x3) conceptually
    # and B as 3x9 = (3x1) * (1x9)
    # The tensor is essentially the "outer" structure.
    #
    # Key structural approach: Use the fact that the multiplication tensor
    # T[i,j,k] = 1 iff (i//m == j%p) and (i%m == j//p)
    # i.e., i//3 == j%9... wait let me re-derive.
    #
    # U index i: 0..5, i = row_A * m + k_A where row_A in {0,1}, k_A in {0,1,2}
    # V index j: 0..26, j = k_B * p + col_B where k_B in {0,1,2}, col_B in {0..8}
    # W index k: 0..17, k = col_B * n + row_A where col_B in {0..8}, row_A in {0,1}
    #
    # T[U_idx, V_idx, W_idx] = 1 iff k_A == k_B (the inner indices match)
    # and row_A matches, col_B matches.
    #
    # So T is a "diagonal" tensor in the inner dimension: for each (row_A, col_B),
    # we need sum over k of A[row_A,k]*B[k,col_B].
    #
    # This is equivalent to: for each (row_A, col_B) pair, we do a dot product of length 3.
    # Total: 2*9 = 18 independent dot products of length 3.
    # Each dot product of length 3 has rank 3 (trivially) or can we do better?
    # No, a single dot product of length 3 has tensor rank exactly 3.
    # But we can share across the 18 dot products!
    #
    # The full tensor: T[a, b, c] where a=(row_A, k_A), b=(k_B, col_B), c=(col_B, row_A)
    # T = 1 iff k_A == k_B.
    #
    # This is a sum over k of (e_k in U-space) * (e_k in V-space) * (all W entries for that k)
    # For fixed k: U has entries at (row_A, k) for row_A in {0,1} -> 2 entries
    # V has entries at (k, col_B) for col_B in {0..8} -> 9 entries
    # W has entries at (col_B, row_A) for all col_B, row_A -> 18 entries
    #
    # So for each k, the "slice" is a 2x9 matrix (outer product of [1,1] and [1,1,...,1])
    # mapped to W indices.
    #
    # Total rank with trivial approach: 3 (one per k), but each "rank-1" term
    # actually needs to cover 2*9=18 output entries.
    #
    # Wait - a rank-1 tensor decomposition term is u * v * w where u, v, w are vectors.
    # For term k: u_k has 1s at positions (0,k) and (1,k) in U-space.
    # v_k has 1s at positions (k,0), (k,1), ..., (k,8) in V-space.
    # w_k has 1s at ALL 18 positions in W-space.
    #
    # So T = sum_{k=0}^{2} u_k * v_k * w_k where w_k is the all-ones vector in W-space!
    #
    # This means rank 3 suffices! Let me verify:
    # (u_k * v_k * w_k)[i, j, l] = u_k[i] * v_k[j] * w_k[l]
    # u_k[i] = 1 iff i = row_A*3 + k for some row_A (i.e., i%3 == k)
    # v_k[j] = 1 iff j = k*9 + col_B for some col_B (i.e., j//9 == k)
    # w_k[l] = 1 for all l
    #
    # So the product is 1 iff i%3 == k AND j//9 == k, i.e., i%3 == j//9.
    # Summing over k: T[i,j,l] = 1 iff i%3 == j//9 (regardless of l).
    #
    # But we need T[i,j,l] = 1 iff i%3 == j//9 AND l = (j%9)*2 + (i//3).
    #
    # So the trivial rank-3 decomposition gives the right "support" in (i,j) but
    # the W component is wrong - it's all-ones instead of the correct pattern.
    #
    # We need: T[i,j,l] = sum_k u_k[i] * v_k[j] * w_k[l]
    # where for the correct (i,j) pair (i%3==j//9==k), we need sum_k w_k[l] = 1 iff l == (j%9)*2 + (i//3).
    #
    # Since u_k[i]=1 and v_k[j]=1 only when k matches, only one term contributes.
    # So we need w_k[l] = 1 iff l == (j%9)*2 + (i//3) where i%3=k, j//9=k.
    # But w_k doesn't depend on i or j! It only depends on k and l.
    #
    # For fixed k, i can be (0,k) or (1,k) [i.e., i = 0*3+k = k or i = 1*3+k = 3+k]
    # j can be (k,0), (k,1), ..., (k,8) [i.e., j = k*9+0, k*9+1, ..., k*9+8]
    # l = (j%9)*2 + (i//3) = (j%9)*2 + row_A
    #
    # So for fixed k, the W entries that should be 1 are:
    # For row_A=0, col_B=0..8: l = col_B*2 + 0 = 0,2,4,6,8,10,12,14,16
    # For row_A=1, col_B=0..8: l = col_B*2 + 1 = 1,3,5,7,9,11,13,15,17
    # That's ALL 18 entries! So w_k IS the all-ones vector.
    #
    # Wait, that means the rank-3 decomposition IS correct!
    # Let me re-verify: T[i,j,l] should be 1 iff i%3 == j//9 AND l == (j%9)*2 + (i//3).
    #
    # My decomposition: sum_k [i%3==k] * [j//9==k] * 1 (for all l)
    # = 1 iff i%3 == j//9, for ALL l.
    #
    # But the target is: 1 iff i%3 == j//9 AND l == (j%9)*2 + (i//3).
    #
    # These are NOT the same! The target has l-dependent structure.
    #
    # So rank 3 is NOT sufficient. The W component must encode the l-dependence.
    #
    # Let me reconsider. We need:
    # T[i,j,l] = sum_r U[i,r] * V[j,r] * W[l,r]
    #
    # For the correct (i,j) pair with i%3==j//9==k:
    # We need sum_r U[i,r]*V[j,r]*W[l,r] = 1 iff l == (j%9)*2 + (i//3)
    #
    # Let's think of this differently. The tensor decomposes as:
    # For each (row_A, col_B) pair, we need a length-3 dot product.
    # There are 2*9 = 18 such dot products.
    #
    # The "natural" decomposition uses rank 3 per dot product, but we can share.
    #
    # Actually, let's think of it as: the tensor is the Kronecker product structure.
    # T = (identity_3 in the k-dimension) tensor (all-ones in the row_A x col_B x l structure)
    #
    # Hmm, let me just try a direct approach with small rank and verify.
    
    # Let's build the target tensor first
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m  # row of A (0 or 1)
        am = i % m   # k index in A (0,1,2)
        for j in range(m*p):
            bj = j // p  # k index in B (0,1,2)
            bm = j % p   # col of B (0..8)
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Strategy: Use the structure to build a small candidate decomposition
    # then use rank reduction to find the minimum.
    
    # Approach 1: Direct construction exploiting structure
    # The tensor T[i,j,k] = 1 iff (i%m == j//p) and k == (j%p)*n + (i//m)
    #
    # Let's try rank = 18 (one per output entry) as starting point,
    # then reduce.
    #
    # Actually, let's try a smarter construction.
    # For each output (col_B, row_A), the contribution is sum_k A[row_A,k]*B[k,col_B].
    # This is a dot product of two length-3 vectors.
    #
    # Rank of a single dot product of length 3 is 3.
    # But across all 18 outputs, we can share the "k" dimension.
    #
    # Construction with rank 3 * 2 = 6? Or rank 3 * 9 = 27?
    #
    # Let me try: for each k in {0,1,2}, create terms that handle the k-th component.
    # Term (k, row_A, col_B): U has 1 at (row_A*3+k), V has 1 at (k*9+col_B), W has 1 at (col_B*2+row_A)
    # That's 3*2*9 = 54 terms (schoolbook).
    #
    # Can we do better? The key is that for fixed k, the structure is:
    # sum over row_A, col_B of [A[row_A,k] * B[k,col_B]] * e_{(col_B,row_A)} in output
    #
    # This is an outer product of A[:,k] (length 2) and B[k,:] (length 9),
    # mapped to the 18-dimensional output space.
    #
    # The rank of this "matrix multiplication" 2x1 * 1x9 -> 2x9 is... 
    # It's just an outer product, which has rank 1!
    #
    # Wait: for fixed k, define a_k = A[:,k] (2-vector), b_k = B[k,:] (9-vector).
    # The contribution to C is a_k * b_k^T (2x9 matrix).
    # As a tensor in the (row_A, col_B, output) space, this is:
    # sum_{row_A, col_B} a_k[row_A] * b_k[col_B] * e_{(col_B,row_A)}
    #
    # This can be written as a single rank-1 tensor:
    # u_k = a_k (in U-space, positions (0,k) and (1,k))
    # v_k = b_k (in V-space, positions (k*9+0) through (k*9+8))
    # w_k = all-ones in W-space
    #
    # But as we showed, w_k = all-ones doesn't work because the target W depends on (row_A, col_B).
    #
    # The issue: the output index l = col_B*2 + row_A couples col_B and row_A.
    # A rank-1 term with w = all-ones would give 1 for ALL l, not just the correct one.
    #
    # So we need to split. For fixed k, we need to represent the 2x9 matrix
    # (with entries 1 at all positions) as a sum of rank-1 terms in the
    # (row_A, col_B) -> l mapping.
    #
    # Actually, the mapping from (row_A, col_B) to l is a bijection (it's just a reindexing).
    # So the "matrix" is actually a 18-dimensional vector of all 1s.
    #
    # The problem is that u_k and v_k are fixed by the input structure, and w_k must
    # encode the output pattern. Since u_k[i]=1 for both row_A values and v_k[j]=1 for
    # all col_B values, the product u_k[i]*v_k[j] = 1 for all (i,j) in the k-slice.
    # So w_k[l] must be 1 for all l. But that's wrong.
    #
    # Resolution: We need to DECOMPOSE the k-slice into multiple rank-1 terms
    # where each term has a different u and v pattern that, combined with w,
    # gives the correct output.
    #
    # Alternative: Use a different factorization.
    #
    # Let me try: rank = 2*9 = 18, one term per output entry.
    # Term for output l = col_B*2 + row_A:
    #   U: has 1 at position row_A*3 + k for all k? No, that doesn't work either.
    #
    # Let me just use a systematic approach:
    # Start with a known valid decomposition and reduce rank.
    
    # Schoolbook: rank 54
    # For each (row_A, k, col_B): U[row_A*3+k]=1, V[k*9+col_B]=1, W[col_B*2+row_A]=1
    rank_school = n * m * p  # 2*3*9 = 54
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    
    col = 0
    for row_A in range(n):
        for k in range(m):
            for col_B in range(p):
                U[row_A * m + k, col] = 1
                V[k * p + col_B, col] = 1
                W[col_B * n + row_A, col] = 1
                col += 1
    
    # Verify schoolbook
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(decomp, T, atol=1e-9), "Schoolbook decomposition failed!"
    
    # Now try to find a smaller decomposition using the structure.
    # 
    # Key insight: The tensor has a "block diagonal" structure in the k-dimension.
    # T[i,j,l] is nonzero only when i%m == j//p.
    #
    # Let's try to use the fact that for each k, the sub-tensor is a "separable" structure.
    #
    # Alternative approach: Use bilinear decomposition directly.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # This is just standard matrix multiplication 2x3 * 3x9.
    # The tensor rank of 2x3*3x9 multiplication...
    #
    # Known results: rank of 2x2*2x2 is 7 (Strassen).
    # rank of 2x3*3x3: at most 7*3 = 21 (block Strassen) or 18 (schoolbook).
    # Actually, 2x3*3x3 can be done in rank 14 (known result).
    # For 2x3*3x9: at most 14*3 = 42 (block) or 54 (schoolbook).
    #
    # But we can be smarter. 2x3*3x9 = 2x3*3*(1x9) ... 
    #
    # Actually, let's think of it as: first compute 2x3*3x3 (giving 2x3), then 2x3*3x9...
    # No, that's not right.
    #
    # Let's try: split the 9 columns into three groups of 3.
    # C = A * [B1 | B2 | B3] where B1, B2, B3 are 3x3.
    # C1 = A*B1, C2 = A*B2, C3 = A*B3, each 2x3.
    #
    # Each 2x3*3x3 multiplication has some rank r.
    # Total rank <= 3r.
    #
    # What's the rank of 2x3*3x3?
    # It's at most 7 (Strassen for 2x2) + 3 (for the extra column) = 10? 
    # Or we can pad 2x3 to 4x4 and use Strassen... 
    #
    # Actually, 2x3*3x3: pad to 4x4 * 4x4, use Strassen (rank 7 per 4x4 block?).
    # No, Strassen is for 2x2*2x2 with rank 7.
    #
    # For 2x3*3x3: we can write it as 2x2*2x2 + 2x1*1x3 + 2x2*2x3...
    # This is getting complicated. Let me just try to find a good decomposition
    # and then reduce.
    
    # Let's try a different structural approach:
    # The tensor T can be written as sum over k of (e_k^U * e_k^V * M_k)
    # where M_k is the 18-dim output vector for the k-th slice.
    # But as we showed, this doesn't directly work with rank-1 terms.
    #
    # Instead, let's use the following:
    # For each k, the contribution is:
    #   sum_{row_A, col_B} A[row_A,k] * B[k,col_B] * delta(l, col_B*2+row_A)
    #
    # We can write delta(l, col_B*2+row_A) as a sum of rank-1 terms in (row_A, col_B, l).
    # The matrix D_k[row_A, col_B, l] = 1 iff l == col_B*2 + row_A.
    #
    # This D is a 2x9x18 tensor. What's its rank?
    # D[row_A, col_B, l] = 1 iff l - row_A == 2*col_B and l - row_A is even.
    #
    # For fixed row_A, D[row_A, :, :] is a 9x18 matrix with 1s at positions (col_B, 2*col_B+row_A).
    # This is a "shifted diagonal" which has rank 9 (each row has one 1).
    #
    # So D has rank at most 2*9 = 18, but likely less due to structure.
    #
    # Actually, D[row_A, col_B, l] = delta(l, 2*col_B + row_A).
    # Let's define for each l, which (row_A, col_B) give 1:
    # l = 2*col_B + row_A, so col_B = (l - row_A)/2, must be integer in 0..8.
    # For l even: row_A=0, col_B=l/2 (if l/2 <= 8)
    # For l odd: row_A=1, col_B=(l-1)/2 (if (l-1)/2 <= 8)
    #
    # So each l corresponds to exactly one (row_A, col_B) pair.
    # D is a "permutation-like" tensor with rank 18 (each output is independent).
    #
    # So the total tensor has rank at most 3 * 18 = 54 (which is schoolbook).
    #
    # Can we do better? The key is whether the 3 k-slices can share rank-1 terms.
    #
    # Let me try a completely different approach: use the known rank of 2x3*3x3.
    #
    # Actually, I recall that the rank of multiplying an m x n matrix by an n x p matrix
    # is at most mn + np + mp - 1 (for some cases) or related to the "tensor rank" bounds.
    #
    # For 2x3*3x9: upper bound from Strassen-like algorithms.
    #
    # Let me try yet another approach: use the fact that 3 = 2+1, and use
    # Strassen for the 2x2 part and handle the 1x1 separately.
    #
    # Split k into {0,1} and {2}:
    # C = A[:,:2]*B[:2,:] + A[:,2]*B[2,:]
    #
    # First term: 2x2 * 2x9. This can be seen as 2x2*2x2 * (identity for the 9/2 split)...
    # Not quite. 2x2*2x9: split the 9 into 4+4+1.
    # 2x2*2x4: use Strassen (rank 7) for each 2x2*2x2 block, so 7*2 = 14 for 2x2*2x4.
    # Plus 2x2*2x1: rank 3 (or 4 with Strassen padding).
    # Total for 2x2*2x9: 14 + 3 = 17? Or 7*2 + 4 = 18?
    #
    # Second term: 2x1 * 1x9: rank 2 (outer product).
    #
    # Total: 17 + 2 = 19? Or 18 + 2 = 20?
    #
    # This is getting too speculative. Let me just implement a systematic search
    # with the schoolbook as starting point and use aggressive rank reduction.
    
    # Better strategy: Build a structured low-rank candidate and verify.
    #
    # Let's try the following construction with rank 18:
    # For each (row_A, col_B) pair, create a term that computes the dot product
    # using the full k-dimension.
    #
    # Term (row_A, col_B): 
    #   U: 1 at positions (row_A*3+0), (row_A*3+1), (row_A*3+2) [all k for this row_A]
    #   V: 1 at positions (0*9+col_B), (1*9+col_B), (2*9+col_B) [all k for this col_B]
    #   W: 1 at position (col_B*2+row_A)
    #
    # Check: sum_{r} U[i,r]*V[j,r]*W[l,r]
    # For term (row_A, col_B): contributes to (i,j,l) iff i//3==row_A, j%9==col_B, l==col_B*2+row_A.
    # The contribution value is U[i,r]*V[j,r]*W[l,r] = 1*1*1 = 1.
    # But we need it to be A[i//3, i%3]*B[j//9, j%9], not 1!
    #
    # This doesn't work because U and V are fixed (0/1) and don't encode the matrix entries.
    #
    # I see the issue: in a tensor decomposition of the MULTIPLICATION tensor,
    # U, V, W are FIXED matrices (not depending on A or B). The decomposition
    # T = sum_r U[:,r] * V[:,r] * W[:,r] means:
    # For any A, B: C = sum_r (U^T * a_r) * (V^T * b_r) * w_r
    # where a_r, b_r are the r-th "bilinear form" applied to A and B.
    #
    # So U[i,r] = coefficient of A[i//3, i%3] in the r-th bilinear form for U.
    # V[j,r] = coefficient of B[j//9, j%9] in the r-th bilinear form for V.
    # W[l,r] = coefficient of C[l%2, l//2] in the r-th output combination.
    #
    # The schoolbook uses U[i,r]=1, V[j,r]=1, W[l,r]=1 for matching (i,j,l).
    #
    # For Strassen-like, we use linear combinations.
    
    # Let me try to construct a rank-18 decomposition using the structure:
    # Split into 3 groups of 3 columns each.
    # For group g (g=0,1,2), we compute A * B[:, 3g:3g+3] (2x3 result).
    #
    # For each group, use a 2x3*3x3 decomposition.
    #
    # 2x3*3x3: Let's use rank 14 (known optimal or near-optimal).
    # Actually, I'm not sure of the exact rank. Let's use rank 15 to be safe.
    #
    # Total: 3 * 15 = 45. Better than 54 but not great.
    #
    # Alternatively, use the (2+1) split of k:
    # C = A[:,:2]*B[:2,:] + A[:,2:3]*B[2:3,:]
    #
    # 2x2*2x9: Split 9 into 4+4+1.
    #   2x2*2x4: two 2x2*2x2 blocks, each rank 7 (Strassen). Total 14.
    #   2x2*2x1: rank 3 (trivial) or 4 (Strassen padded). Use 3.
    #   Total for 2x2*2x9: 14 + 3 = 17.
    #
    # 2x1*1x9: rank 2 (outer product, one term per output row).
    #
    # Grand total: 17 + 2 = 19.
    #
    # Let me implement this rank-19 construction!
    
    rank19 = 19
    U19 = np.zeros((n*m, rank19))
    V19 = np.zeros((m*p, rank19))
    W19 = np.zeros((p*n, rank19))
    
    col = 0
    # Strassen for 2x2*2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5-m7
    # c12 = m3+m5+m6
    # c21 = m2+m4+m7
    # c22 = m1-m2+m3+m6
    
    # For 2x2*2x4 (two 2x2*2x2 blocks):
    # Block 1: columns 0,1 of B (for the 2x2*2x2)
    # Block 2: columns 2,3 of B
    
    for block in range(2):  # two 2x2*2x2 blocks
        # B columns for this block: 2*block, 2*block+1
        b_col0 = 2 * block
        b_col1 = 2 * block + 1
        
        # Strassen terms
        # m1 = (a11+a22)(b11+b22)
        # In our indexing: A is 2x2 (rows 0,1; cols 0,1), B is 2x2 (rows 0,1; cols b_col0, b_col1)
        # U index for A[i,k]: i*2+k (but we're in the 2x2 sub-problem, so i in 0,1, k in 0,1)
        # Full U index: i*m + k = i*3 + k
        # Full V index: k*p + j = k*9 + j
        # Full W index: j*n + i = j*2 + i
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,b_col0]+B[1,b_col1])
        U19[0*3+0, col] += 1  # A[0,0]
        U19[1*3+1, col] += 1  # A[1,1]
        V[0*9+b_col0, col] += 1  # B[0, b_col0]
        V[1*9+b_col1, col] += 1  # B[1, b_col1]
        # Output: c[0,0] += m1, c[0,1] += m1, c[1,0] += m1, c[1,1] += m1
        # In full W: c[row, col] = W[col*2+row]
        W19[b_col0*2+0, col] += 1   # c[0, b_col0]
        W19[b_col1*2+0, col] += 1   # c[0, b_col1]
        W19[b_col0*2+1, col] += 1   # c[1, b_col0]
        W19[b_col1*2+1, col] += 1   # c[1, b_col1]
        col += 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,b_col0]
        U19[1*3+0, col] += 1  # A[1,0]
        U19[1*3+1, col] += 1  # A[1,1]
        V[0*9+b_col0, col] += 1
        W19[b_col0*2+1, col] += 1   # c[1, b_col0]
        col += 1
        
        # m3 = A[0,0] * (B[0,b_col1]-B[1,b_col1])
        U19[0*3+0, col] += 1
        V[0*9+b_col1, col] += 1
        V[1*9+b_col1, col] -= 1
        W19[b_col1*2+0, col] += 1   # c[0, b_col1]
        col += 1
        
        # m4 = A[1,1] * (B[1,b_col0]-B[0,b_col0])
        U19[1*3+1, col] += 1
        V[1*9+b_col0, col] += 1
        V[0*9+b_col0, col] -= 1
        W19[b_col0*2+1, col] += 1   # c[1, b_col0]
        col += 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1,b_col1]
        U19[0*3+0, col] += 1
        U19[0*3+1, col] += 1
        V[1*9+b_col1, col] += 1
        W19[b_col1*2+0, col] += 1   # c[0, b_col1]
        col += 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,b_col0]+B[0,b_col1])
        U19[1*3+0, col] += 1
        U19[0*3+0, col] -= 1
        V[0*9+b_col0, col] += 1
        V[0*9+b_col1, col] += 1
        W19[b_col1*2+0, col] += 1   # c[0, b_col1]
        col += 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,b_col0]+B[1,b_col1])
        U19[0*3+1, col] += 1
        U19[1*3+1, col] -= 1
        V[1*9+b_col0, col] += 1
        V[1*9+b_col1, col] += 1
        W19[b_col1*2+1, col] += 1   # c[1, b_col1]
        col += 1
    
    # Handle columns 4,5,6,7,8 of B (5 more columns)
    # Use 2x2*2x1 for each pair, plus 2x2*2x1 for the last one
    # Actually, let's use 2x2*2x1 for columns 4&5, 6&7, and 2x2*2x1 for column 8
    
    # For 2x2*2x1 (single output column j):
    # c[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # c[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    # Rank 3:
    # m1 = (A[0,0]+A[1,1])*(B[0,j]+B[1,j])
    # m2 = (A[0,0]-A[1,1])*(B[0,j]-B[1,j])
    # c[0,j] = (m1+m2)/2 + A[0,1]*B[1,j]... this is getting messy.
    #
    # Just use schoolbook for the remaining 5 columns: 2*5 = 10 terms
    # Total so far: 14 (from 2 blocks of Strassen) + 10 = 24. Hmm, that's more than 19.
    #
    # Let me reconsider. Maybe just use schoolbook for the 2x2*2x9 part: 2*2*9 = 36 terms.
    # Plus 2x1*1x9: 2 terms. Total 38.
    #
    # That's better than 54 but not great. Let me try a different split.
    
    # Actually, let me just use the schoolbook (rank 54) as starting point
    # and apply aggressive rank reduction. The current program does this
    # but with a rank-50 starting point. Let me try with rank 54 and
    # a more aggressive search.
    
    # Reset to schoolbook
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for row_A in range(n):
        for k in range(m):
            for col_B in range(p):
                U[row_A * m + k, col] = 1
                V[k * p + col_B, col] = 1
                W[col_B * n + row_A, col] = 1
                col += 1
    
    # Verify
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    assert np.allclose(decomp, T, atol=1e-9)
    
    # Build active indices
    active_ijk = []
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                if T[i,j,k] != 0:
                    active_ijk.append((i,j,k))
    
    N_act = len(active_ijk)
    
    # For rank reduction, we need to find the minimum number of terms.
    # This is a hard problem, but we can use greedy methods.
    #
    # Key insight: many schoolbook terms may be linearly dependent in the
    # "tensor product" sense. We can try to find a subset of terms that
    # still spans the target.
    
    # Build the constraint matrix
    M_mat = np.zeros((N_act, rank_school))
    for idx, (i, j, k) in enumerate(active_ijk):
        for r in range(rank_school):
            M_mat[idx, r] = U[i, r] * V[j, r] * W[k, r]
    
    b_target = np.ones(N_act)
    
    # Verify
    assert np.allclose(M_mat @ np.ones(rank_school), b_target, atol=1e-9)
    
    # Greedy rank reduction: try to remove terms one by one
    # A term r can be removed if b_target is in the column space of M_mat[:, remaining]
    
    def can_represent(cols, M_mat, b_target):
        if len(cols) == 0:
            return False
        A = M_mat[:, cols]
        try:
            c, residuals, rank_s, sv = np.linalg.lstsq(A, b_target, rcond=1e-12)
            res = np.linalg.norm(A @ c - b_target)
            return res < 1e-10
        except Exception:
            return False
    
    def verify_decomp(U_mat, V_mat, W_mat, T_target):
        decomp = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(decomp, T_target, atol=1e-9)
    
    best_mask = np.ones(rank_school, dtype=bool)
    best_rank_val = rank_school
    
    # Multiple passes of greedy removal
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for r in range(rank_school):
            if time.time() > deadline - 10:
                break
            if not best_mask[r]:
                continue
            new_mask = best_mask.copy()
            new_mask[r] = False
            cols = np.where(new_mask)[0]
            if can_represent(cols, M_mat, b_target):
                U_new = U[:, new_mask]
                V_new = V[:, new_mask]
                W_new = W[:, new_mask]
                if verify_decomp(U_new, V_new, W_new, T):
                    best_mask = new_mask
                    best_rank_val = int(new_mask.sum())
                    improved = True
    
    if best_rank_val < rank_school:
        U_out = U[:, best_mask]
        V_out = V[:, best_mask]
        W_out = W[:, best_mask]
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # If no improvement found, try a different starting decomposition
    # Use the Strassen-based rank-50 from the original program
    
    rank50 = 50
    U50 = np.zeros((n*m, rank50))
    V50 = np.zeros((m*p, rank50))
    W50 = np.zeros((p*n, rank50))
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U50[u_a][col] += 1
        U50[u_d][col] += 1
        V50[v_e][col] += 1
        V50[v_h][col] += 1
        
        U50[u_c][col+1] += 1
        U50[u_d][col+1] += 1
        V50[v_e][col+1] += 1
        
        U50[u_a][col+2] += 1
        V50[v_f][col+2] += 1
        V50[v_h][col+2] -= 1
        
        U50[u_d][col+3] += 1
        V50[v_g][col+3] += 1
        V50[v_e][col+3] -= 1
        
        U50[u_a][col+4] += 1
        U50[u_b][col+4] += 1
        V50[v_h][col+4] += 1
        
        U50[u_c][col+5] += 1
        U50[u_a][col+5] -= 1
        V50[v_e][col+5] += 1
        V50[v_f][col+5] += 1
        
        U50[u_b][col+6] += 1
        U50[u_d][col+6] -= 1
        V50[v_g][col+6] += 1
        V50[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W50[w_00][col] += 1
        W50[w_00][col+3] += 1
        W50[w_00][col+4] -= 1
        W50[w_00][col+6] += 1
        
        W50[w_01][col+2] += 1
        W50[w_01][col+4] += 1
        
        W50[w_10][col+1] += 1
        W50[w_10][col+3] += 1
        
        W50[w_11][col] += 1
        W50[w_11][col+1] -= 1
        W50[w_11][col+2] += 1
        W50[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U50[u_a][col] += 1
    V50[v_e][col] += 1
    U50[u_b][col+1] += 1
    V50[v_g][col+1] += 1
    U50[u_c][col+2] += 1
    V50[v_e][col+2] += 1
    U50[u_d][col+3] += 1
    V50[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W50[w_08][col] += 1
    W50[w_08][col+1] += 1
    W50[w_18][col+2] += 1
    W50[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U50[u_ar0][col] += 1
        V50[v_b][col] += 1
        W50[k * n + 0][col] += 1
        U50[u_ar1][col+1] += 1
        V50[v_b][col+1] += 1
        W50[k * n + 1][col+1] += 1
        col += 2
    
    # Verify rank-50
    if np.allclose(np.einsum('ir,jr,kr->ijk', U50, V50, W50), T, atol=1e-9):
        # Try rank reduction on rank-50
        M_mat50 = np.zeros((N_act, rank50))
        for idx, (i, j, k) in enumerate(active_ijk):
            for r in range(rank50):
                M_mat50[idx, r] = U50[i, r] * V50[j, r] * W50[k, r]
        
        best_mask50 = np.ones(rank50, dtype=bool)
        best_rank50 = rank50
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for r in range(rank50):
                if time.time() > deadline - 5:
                    break
                if not best_mask50[r]:
                    continue
                new_mask = best_mask50.copy()
                new_mask[r] = False
                cols = np.where(new_mask)[0]
                if can_represent(cols, M_mat50, b_target):
                    U_new = U50[:, new_mask]
                    V_new = V50[:, new_mask]
                    W_new = W50[:, new_mask]
                    if verify_decomp(U_new, V_new, W_new, T):
                        best_mask50 = new_mask
                        best_rank50 = int(new_mask.sum())
                        improved = True
        
        if best_rank50 < best_rank_val:
            U_out = U50[:, best_mask50]
            V_out = V50[:, best_mask50]
            W_out = W50[:, best_mask50]
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    # Return the best found
    if best_rank_val < rank_school:
        U_out = U[:, best_mask]
        V_out = V[:, best_mask]
        W_out = W[:, best_mask]
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
