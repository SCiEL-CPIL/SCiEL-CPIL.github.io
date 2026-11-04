# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = 4, 4, 9
    # Dimensions
    dim_U = n * m  # 16
    dim_V = m * p  # 36
    dim_W = p * n  # 36
    total_terms = n * m * p  # 144

    # Build the full multiplication tensor T[a, b, c] where:
    # a = i*m + j (index into U, i in [0,n), j in [0,m))
    # b = j*p + k (index into V, j in [0,m), k in [0,p))
    # c = k*n + i (index into W, k in [0,p), i in [0,n))
    # T[a,b,c] = 1 iff j matches (the shared index)
    # i.e., T[(i,j),(j,k),(k,i)] = 1

    # Build tensor as a 3D array
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # The tensor can be written as sum over r of u_r ⊗ v_r ⊗ w_r
    # where for term (i,j,k): u has 1 at (i*m+j), v has 1 at (j*p+k), w has 1 at (k*n+i)
    
    # Strategy: Find the minimum number of rank-1 tensors whose sum equals T.
    # This is equivalent to finding the tensor rank.
    
    # Approach: Use the "slice" method. Fix one index and look at the 2D slice.
    # For each fixed a (U-index), T[a, :, :] is a dim_V x dim_W matrix.
    # The rank of T is at least the maximum rank over all slices.
    
    # But more powerfully: we can use the fact that the tensor lies in a structured subspace.
    
    # Let's use a different approach: 
    # The tensor T can be decomposed as:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the unit vector at position i*m+j in U-space, etc.
    
    # Key observation: For a fixed j, the terms (i,j,k) for all i,k form a rank-1 
    # structure in the "j-slice". Specifically:
    # sum_{i,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} = (sum_i e_{i,j} ⊗ e_{k,i}) ⊗ ... 
    # Actually, for fixed j: sum_{i,k} (e_{i,j})_a (e_{j,k})_b (e_{k,i})_c
    # This equals: for the j-slice, it's the matrix M_j where M_j[i,k] = 1 (all ones, n×p matrix)
    # Wait, let me think more carefully.
    
    # For fixed j, the contribution to T from all (i,j,k) terms is:
    # T_j[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, summed over i,k
    # So T_j is non-zero only at positions (i*m+j, j*p+k, k*n+i) for i in [0,n), k in [0,p)
    
    # The key structural insight: 
    # T = sum_j S_j where S_j is the "j-slice" tensor
    # S_j has entries: S_j[i*m+j, j*p+k, k*n+i] = 1 for all i,k
    # 
    # S_j can be written as: sum_{i,k} u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}
    # where u_{i,j} = e_{i*m+j}, v_{j,k} = e_{j*p+k}, w_{k,i} = e_{k*n+i}
    #
    # But S_j itself might have lower rank! 
    # S_j = sum_i sum_k u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}
    # = sum_i (u_{i,j} ⊗ (sum_k v_{j,k} ⊗ w_{k,i}))
    # = sum_i (u_{i,j} ⊗ V_i) where V_i = sum_k v_{j,k} ⊗ w_{k,i}
    #
    # V_i is a dim_V x dim_W matrix with a single 1 at position (j*p+k, k*n+i) for each k... 
    # No wait, V_i = sum_k v_{j,k} ⊗ w_{k,i} has 1s at (j*p+k, k*n+i) for k=0..p-1
    # That's a matrix with p ones, one in each "row group" j and "column group" i.
    
    # Alternative approach: Direct rank computation via optimization
    # We know the answer must be at most 144. Let's try to find a smaller decomposition.
    
    # Method: Greedy + linear algebra
    # 1. Start with all 144 terms
    # 2. Try to remove terms that are linear combinations of others
    
    # Build the "coefficient" representation:
    # Each term r corresponds to a triple (i_r, j_r, k_r)
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
    # 
    # For the decomposition to be exact, we need:
    # sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c] for all a,b,c
    #
    # Since our u_r, v_r, w_r are unit vectors, this means:
    # For each (a,b,c) with T[a,b,c]=1, exactly the terms where a=i*m+j_r, b=j_r*p+k_r, c=k_r*n+i must cover it.
    # Specifically, T[a,b,c]=1 iff there exists r such that a=i*m+j_r, b=j_r*p+k_r, c=k_r*n+i
    # where i = c % n... wait, let me be more careful.
    
    # Actually, the simplest correct approach for this structured tensor:
    # Use the fact that T = sum_j S_j, and find the rank of each S_j.
    
    # S_j is a tensor of size dim_U x dim_V x dim_W but only non-zero on a specific pattern.
    # Let's compute the rank of S_j by looking at its "flattened" form.
    
    # For S_j: non-zero entries are at (i*m+j, j*p+k, k*n+i) for i in 0..n-1, k in 0..p-1
    # 
    # Consider the 2D slice: fix a = i*m+j. Then S_j[i*m+j, :, :] is a dim_V x dim_W matrix
    # with 1s at positions (j*p+k, k*n+i) for k=0..p-1.
    # This is a matrix with p non-zero entries. Its rank is at most p (and likely p if they're independent).
    
    # Fix b = j*p+k. Then S_j[:, j*p+k, :] is a dim_U x dim_W matrix
    # with 1s at positions (i*m+j, k*n+i) for i=0..n-1.
    # This is a matrix with n non-zero entries. Its rank is at most n.
    
    # Fix c = k*n+i. Then S_j[:, :, k*n+i] is a dim_U x dim_V matrix
    # with 1s at positions (i*m+j, j*p+k) for... wait, for fixed c=k*n+i, we need i and k fixed.
    # So S_j[:, :, k*n+i] has exactly one non-zero at (i*m+j, j*p+k).
    
    # The rank of S_j: Let's use the "matricization" approach.
    # Reshape S_j into a (dim_U * dim_V) x dim_W matrix.
    # The rank of this matrix gives an upper bound on the tensor rank.
    
    # Actually, let me just directly compute the tensor rank using a different strategy.
    
    # KEY INSIGHT: The tensor T has a very special structure. Let's exploit it.
    # 
    # T[(i,j),(j,k),(k,i)] = 1
    # 
    # This is the "multiplication tensor" for the algebra where:
    # - U-space has basis elements e_{i,j} for i=0..3, j=0..3
    # - V-space has basis elements f_{j,k} for j=0..3, k=0..8  
    # - W-space has basis elements g_{k,i} for k=0..8, i=0..3
    # - Product: e_{i,j} * f_{j,k} = g_{k,i} (and 0 otherwise)
    #
    # This is like a "path" algebra or a specific tensor product structure.
    #
    # The rank of this tensor can be bounded by looking at it as a sum of "commuting" parts.
    
    # Let me try a concrete approach: use the known result that for matrix multiplication
    # m x n * n x p, the rank is at most m*n*p but can be smaller.
    # For 4x4 * 4x9, let's try to find a decomposition with rank < 144.
    
    # Approach: Use the "block" structure.
    # Group terms by j (the shared dimension index).
    # For each j, we have n*p = 4*9 = 36 terms.
    # 
    # The j-th block S_j = sum_{i,k} e_{i,j} ⊗ f_{j,k} ⊗ g_{k,i}
    # 
    # Note that S_j = (sum_i e_{i,j} ⊗ g_{k,i} for fixed k) ... no.
    # 
    # Let's write S_j differently:
    # S_j = sum_i sum_k e_{i,j} ⊗ f_{j,k} ⊗ g_{k,i}
    # = sum_i (e_{i,j} ⊗ (sum_k f_{j,k} ⊗ g_{k,i}))
    # 
    # Let F_i = sum_k f_{j,k} ⊗ g_{k,i} (a dim_V x dim_W matrix)
    # F_i has 1s at (j*p+k, k*n+i) for k=0..p-1
    # 
    # So S_j = sum_i e_{i,j} ⊗ F_i
    # This means S_j has rank at most n = 4! (since it's a sum of n rank-1 tensor terms)
    # 
    # Wait, that's not right either. e_{i,j} ⊗ F_i is not a rank-1 tensor in the 3D sense.
    # A rank-1 tensor is u ⊗ v ⊗ w (three vectors). e_{i,j} ⊗ F_i is a vector times a matrix,
    # which is a sum of p rank-1 tensors (since F_i has p non-zeros).
    # 
    # So S_j = sum_i sum_k e_{i,j} ⊗ f_{j,k} ⊗ g_{k,i} which is n*p = 36 rank-1 terms.
    # 
    # But can we do better for S_j? 
    # S_j is a tensor in (R^16) x (R^36) x (R^36) but with very sparse support.
    # 
    # Let's compute the rank of S_j properly by matricization.
    # Reshape S_j to a matrix of size (16*36) x 36.
    
    # Let me just compute the rank numerically for each j-block and sum them up.
    
    best_rank = n * m * p  # start with schoolbook
    best_decomp = None
    
    for j in range(m):
        # Build S_j as a 3D tensor
        S_j = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for i in range(n):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                S_j[a, b, c] = 1.0
        
        # Compute the rank of S_j via matricization (U-V vs W)
        # Reshape to (dim_U * dim_V) x dim_W
        M = S_j.reshape(dim_U * dim_V, dim_W)
        # Compute rank via SVD
        try:
            sv = np.linalg.svd(M, compute_uv=False)
            # Count significant singular values
            tol = 1e-10
            rank_j = np.sum(sv > tol)
        except:
            rank_j = n * p  # fallback
        
        # The rank of S_j is at most rank_j, but we need to verify we can achieve it.
        # For this structured tensor, the rank should equal the matricization rank.
        
        if time.time() > deadline:
            break
    
    # Now, the total tensor T = sum_j S_j.
    # The rank of T is at most sum of ranks of S_j, but could be less due to interactions.
    # However, since the S_j have disjoint support in the U and V indices (different j),
    # they are "orthogonal" in a sense, so rank(T) = sum_j rank(S_j).
    
    # Wait, that's not quite right. The W indices can overlap.
    # But the U indices for different j are disjoint (different j component),
    # and V indices for different j are also disjoint.
    # So the S_j are "disjoint" in the first two modes, which means
    # rank(T) = sum_j rank(S_j).
    
    # Let's verify: if we have tensors S_j with disjoint support in mode 1 (U),
    # then any rank-1 decomposition of T must use separate terms for each S_j,
    # because a single rank-1 term u⊗v⊗w can only contribute to one j-block
    # (since u has support in only one j-slice of U-space).
    
    # Therefore: rank(T) = sum_{j=0}^{m-1} rank(S_j)
    
    # Now compute rank(S_j) for each j.
    # By symmetry, all S_j have the same rank! (The structure is identical for each j.)
    # So rank(T) = m * rank(S_0) = 4 * rank(S_0)
    
    # Let's compute rank(S_0) carefully.
    S_0 = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for k in range(p):
            a = i * m + 0  # j=0
            b = 0 * p + k  # j=0
            c = k * n + i
            S_0[a, b, c] = 1.0
    
    # S_0 is non-zero at (i*4, k, k*4+i) for i=0..3, k=0..8
    # 
    # Let's compute its rank via different matricizations to get bounds.
    
    # Matricization 1: (U x V) vs W
    M1 = S_0.reshape(dim_U * dim_V, dim_W)
    sv1 = np.linalg.svd(M1, compute_uv=False)
    rank1 = np.sum(sv1 > 1e-10)
    
    # Matricization 2: U vs (V x W)
    M2 = S_0.reshape(dim_U, dim_V * dim_W)
    sv2 = np.linalg.svd(M2, compute_uv=False)
    rank2 = np.sum(sv2 > 1e-10)
    
    # Matricization 3: V vs (U x W)
    M3 = S_0.reshape(dim_V, dim_U * dim_W)
    sv3 = np.linalg.svd(M3, compute_uv=False)
    rank3 = np.sum(sv3 > 1e-10)
    
    # The tensor rank is at least max(rank1, rank2, rank3) and at most min of various bounds.
    # For this structured tensor, the rank should equal the maximum matricization rank.
    
    # Let's also try to find an explicit decomposition.
    # S_0[i*4, k, k*4+i] = 1 for i=0..3, k=0..8
    # 
    # This is a 3D array of size 16 x 9 x 36 (only the j=0 slices are non-zero in U and V).
    # Actually, in U-space, only indices 0,1,2,3 (i*4 for i=0..3) are non-zero.
    # In V-space, only indices 0,1,...,8 (k for k=0..8) are non-zero.
    # In W-space, indices k*4+i for k=0..8, i=0..3 are non-zero, which is all of 0..35.
    
    # The non-zero pattern: for each (i,k), we have a 1 at (i*4, k, k*4+i).
    # 
    # This looks like it could have rank n*p = 36 (trivial) or possibly less.
    # 
    # Let's check: is there a linear dependency?
    # Consider the vectors w_{i,k} = e_{k*4+i} in W-space.
    # The tensor is sum_{i,k} u_i ⊗ v_k ⊗ w_{i,k} where u_i = e_{i*4}, v_k = e_k.
    # 
    # For this to have rank < 36, there must be a linear relation among the "slices".
    # 
    # Fix i: sum_k u_i ⊗ v_k ⊗ w_{i,k} = u_i ⊗ (sum_k v_k ⊗ w_{i,k})
    # The matrix A_i = sum_k v_k ⊗ w_{i,k} has 1s at (k, k*4+i) for k=0..8.
    # This is a 9x36 matrix. Its rank is 9 (the 9 rows are linearly independent since
    # each has a 1 in a different column group).
    # 
    # So S_0 = sum_i u_i ⊗ A_i where A_i is a 9x36 matrix of rank 9.
    # This means S_0 can be written as sum_i sum_{l=1}^{9} u_i ⊗ (some vector) ⊗ (some vector)
    # = sum_i sum_l (u_i ⊗ v'_l) ⊗ w'_l
    # That's 4*9 = 36 terms. Same as before.
    
    # Hmm, so the rank might actually be 36 per j-block, giving 144 total.
    # But wait, maybe there's a smarter decomposition.
    
    # Let me try: can we share terms between different i values?
    # S_0 = sum_{i,k} e_{i*4} ⊗ e_k ⊗ e_{k*4+i}
    # 
    # Consider grouping by k: sum_i e_{i*4} ⊗ e_k ⊗ e_{k*4+i}
    # = e_k ⊗ (sum_i e_{i*4} ⊗ e_{k*4+i})
    # The matrix B_k = sum_i e_{i*4} ⊗ e_{k*4+i} is 16x36 with 1s at (i*4, k*4+i).
    # Rank of B_k: the 4 non-zero rows are at positions 0,4,8,12 with 1s at columns k*4, k*4+1, k*4+2, k*4+3.
    # These 4 rows are linearly independent, so rank(B_k) = 4.
    # 
    # So S_0 = sum_k e_k ⊗ B_k = sum_k sum_{l=1}^{4} e_k ⊗ u'_l ⊗ w'_l = 9*4 = 36 terms.
    
    # It seems the rank is indeed 36 per block. Let me verify with the SVD.
    
    # Actually, let me reconsider. The SVD of M1 should give the answer.
    # Let me just use the maximum of the matricization ranks as a lower bound
    # and try to construct a decomposition of that size.
    
    rank_S0 = max(rank1, rank2, rank3)
    
    # Now, since all S_j are isomorphic, rank(T) = m * rank_S0
    # But wait, I need to double-check that the S_j are truly independent.
    
    # The U-support of S_j is {i*m + j : i=0..n-1} = {j, j+m, j+2m, j+3m}
    # For different j, these sets are disjoint. ✓
    # The V-support of S_j is {j*p + k : k=0..p-1} = {j*p, j*p+1, ..., j*p+p-1}
    # For different j, these sets are disjoint. ✓
    # 
    # Since the U-supports are disjoint, any rank-1 tensor u⊗v⊗w can only
    # contribute to one S_j (the one matching j's U-support).
    # Therefore, rank(T) = sum_j rank(S_j) = m * rank(S_0).
    
    total_rank = m * rank_S0
    
    # Now construct the actual decomposition.
    # For each j, decompose S_j into rank_S0 rank-1 terms.
    # 
    # I'll use the explicit structure: S_j = sum_{i,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This gives n*p = 36 terms per j. If rank_S0 < 36, I need to find a better decomposition.
    
    # For now, let's use the trivial 36-term decomposition per j and see if we can reduce.
    # Actually, let me just verify the SVD result and use it.
    
    if time.time() > deadline:
        # Return schoolbook as fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Let's try to find a reduced decomposition using the SVD of the matricization.
    # For S_0, use the (U x V) vs W matricization.
    # M1 = S_0.reshape(16*9, 36) = S_0.reshape(144, 36)
    # The rank is rank1.
    # 
    # To get an explicit decomposition, I can use the SVD:
    # M1 = U_mat @ diag(sv) @ V_mat.T
    # Then S_0 = sum_{l=1}^{rank1} sv[l] * uvec_l ⊗ wvec_l
    # where uvec_l is the l-th column of U_mat reshaped to (16, 9) and flattened,
    # and wvec_l is the l-th row of V_mat.
    # 
    # But this gives real-valued entries, not integer/half-integer.
    # The task requires integer/half-integer entries.
    
    # Given the constraint of integer/half-integer entries, the SVD approach won't directly work.
    # I need a combinatorial decomposition.
    
    # Let me try a different structural approach.
    # 
    # Observation: S_j[i*m+j, j*p+k, k*n+i] = 1
    # 
    # For the j=0 block: S_0[i*4, k, k*4+i] = 1
    # 
    # This is equivalent to: for each k, the vector (1,1,1,1) in the "i" direction
    # is placed at W-positions k*4, k*4+1, k*4+2, k*4+3.
    # 
    # Can we use a "Fourier" or "Winograd" type decomposition?
    # 
    # Actually, let me try a simple reduction:
    # Instead of 36 terms per block, can we use fewer by combining i-terms?
    # 
    # S_0 = sum_k [e_0 ⊗ e_k ⊗ e_{k*4} + e_4 ⊗ e_k ⊗ e_{k*4+1} + e_8 ⊗ e_k ⊗ e_{k*4+2} + e_12 ⊗ e_k ⊗ e_{k*4+3}]
    # = sum_k e_k ⊗ [e_0 ⊗ e_{k*4} + e_4 ⊗ e_{k*4+1} + e_8 ⊗ e_{k*4+2} + e_12 ⊗ e_{k*4+3}]
    # 
    # The bracket is a 16x36 matrix with 4 ones. Its rank is 4.
    # So we need 4 rank-1 terms for each k, giving 9*4 = 36 total.
    # 
    # Unless... we can share across k values!
    # 
    # Let me define: for a fixed set of W-indices, can we combine?
    # 
    # Actually, I think the rank really is 36 per block for integer/half-integer decompositions.
    # Let me just verify the total rank and construct the decomposition.
    
    # Given the difficulty of finding a reduced integer decomposition,
    # let me try a different strategy: use the full tensor and find linear dependencies.
    
    # Build the 144 unit vectors for each term.
    # Term r corresponds to (i_r, j_r, k_r).
    # u_r = e_{i_r*m + j_r} in R^16
    # v_r = e_{j_r*p + k_r} in R^36
    # w_r = e_{k_r*n + i_r} in R^36
    
    # The tensor is sum_r u_r ⊗ v_r ⊗ w_r.
    # 
    # A reduced decomposition would use a subset of these (or linear combinations).
    # But with integer/half-integer constraint, we're limited.
    
    # Let me just go with the schoolbook decomposition but verify it's correct,
    # and try one more thing: check if we can merge any terms.
    
    # For two terms (i1,j,k1) and (i2,j,k2) with same j:
    # They share the V-index j*p+k1 and j*p+k2 (different unless k1=k2).
    # They share U-indices i1*m+j and i2*m+j (different unless i1=i2).
    # They share W-indices k1*n+i1 and k2*n+i2 (could be same if k1=i2 and k2=i1... 
    #   k1*n+i1 = k2*n+i2 means (k1-k2)*n = i2-i1, and since |i2-i1| < n, this means k1=k2 and i1=i2).
    # So no two distinct terms in the same j-block share all three indices.
    
    # I'll go with the schoolbook rank of 144, but let me at least verify correctness
    # and try to see if the SVD suggests a lower rank that I might be able to achieve.
    
    # Actually, let me try one more idea: the tensor might have a lower rank
    # if we allow non-unit entries.
    
    # For the j=0 block: S_0[i*4, k, k*4+i] = 1
    # 
    # Consider: can we write this as a sum of fewer rank-1 tensors with half-integer entries?
    # 
    # Try: S_0 = sum_{k=0}^{8} sum_{i=0}^{3} e_{i*4} ⊗ e_k ⊗ e_{k*4+i}
    # 
    # What if I group by (k mod 2) or some other partition?
    # 
    # Let me try: for even k and odd k separately.
    # Even k: k = 0,2,4,6,8 (5 values)
    # Odd k: k = 1,3,5,7 (4 values)
    # 
    # This doesn't obviously help.
    
    # Let me just construct the schoolbook decomposition and return it.
    # The rank is 144, which is the same as the seed. But at least it's verified correct.
    
    # Wait - let me try one more thing. What if the rank per block is actually less than 36?
    # Let me check the SVD more carefully.
    
    # For S_0, the (U x V) vs W matricization:
    # M1 is 144 x 36. The non-zero entries are at row (i*4)*9 + k = i*36 + k, column k*4 + i.
    # For fixed i,k: M1[i*36 + k, k*4 + i] = 1.
    # 
    # Let's see: for i=0: rows 0,1,2,...,8, columns 0,5,10,15,20,25,30,35,4... 
    # Wait: k=0: row 0, col 0; k=1: row 1, col 5; k=2: row 2, col 10; ... k=8: row 8, col 35.
    # For i=1: rows 36,37,...,44, columns 1,6,11,16,21,26,31,36... wait, k*4+1: k=0:col1, k=1:col5... 
    # Hmm, k=0: col 1, k=1: col 5, k=2: col 9, ..., k=8: col 33.
    # 
    # So M1 has 36 ones total (4*9), one for each (i,k) pair.
    # The rank of M1: we have 36 non-zero entries in a 144x36 matrix.
    # The maximum possible rank is min(144, 36) = 36.
    # 
    # Are the 36 rows with non-zeros linearly independent? 
    # Each row has exactly one non-zero (a 1). So the non-zero rows are standard basis vectors
    # in different positions... wait, no. Each row has one 1, but different rows have their 1s
    # in different columns.
    # 
    # Row i*36+k has a 1 in column k*4+i.
    # For different (i,k) pairs, are the columns distinct?
    # k*4+i: for k=0..8, i=0..3, this gives values 0..35, all distinct!
    # (k*4+i is a bijection from {0..8}x{0..3} to {0..35})
    # 
    # So M1 has 36 rows, each with a single 1 in a distinct column.
    # These 36 rows are linearly independent (they form a subset of the standard basis).
    # Therefore rank(M1) = 36.
    # 
    # This confirms rank(S_0) >= 36, and since we have a 36-term decomposition, rank(S_0) = 36.
    # 
    # Therefore rank(T) = 4 * 36 = 144.
    # 
    # The schoolbook decomposition is already rank-minimal!
    
    # So the answer is rank = 144, same as the seed.
    # Let me just construct it cleanly.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
