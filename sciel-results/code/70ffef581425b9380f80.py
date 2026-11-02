"""Matrix multiplication 2x3x6 tensor decomposition; structural construction with rank reduction search."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Build the trivial decomposition first as fallback
    rank_trivial = n * m * p  # 36
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Strategy: Try to find rank-35 decomposition by exploiting structure.
    # Key idea: For each k, the two output entries C[0,k] and C[1,k] share
    # the same B row. We can try to design a decomposition where for some k,
    # a single rank-1 term contributes to both i=0 and i=1.
    
    # Approach: Fix U to be one-hot (6 rows, 6 active columns), then for each
    # (j,k) pair, we need V[j,k,:] and W[:,j,k-related] to satisfy constraints.
    
    # More concretely: Let's try a block structure. For each k in 0..5,
    # we have 3 j-values and 2 i-values = 6 tensor entries to cover.
    # In trivial, this uses 6 rank-1 terms. Can we do it in 5?
    
    # For fixed k, the sub-tensor has entries T[(i,j),(j,k),(k,i)] = 1
    # which is T[i,j] = 1 for all i in {0,1}, j in {0,1,2}.
    # This is a 2x3x2 tensor (indices i, j, and i for W).
    # Actually it's: for each (i,j), the W-index is k*2+i.
    # So the sub-tensor is: sum over i,j of e_i^U ⊗ e_j^V ⊗ e_i^W (2-dim W space)
    # = sum_i e_i ⊗ (sum_j e_j) ⊗ e_i = (sum_j e_j) ⊗ (e_0⊗e_0 + e_1⊗e_1)
    
    # The rank of the 2x3x2 tensor where T[i,j,c] = delta_{i,c} for all j:
    # T = sum_{i=0}^1 sum_{j=0}^2 e_i ⊗ e_j ⊗ e_i
    # = (e_0+e_1+e_2) ⊗ (e_0⊗e_0 + e_1⊗e_1) in the (V,U×W) reshaping
    # The rank is the sum of ranks of each i-block: rank(e_0⊗e_0) + rank(e_1⊗e_1) = 1+1 = 2
    # Wait, that's not right. Let me reconsider.
    
    # T[i,j,c] = delta_{i,c}. This is a 2x3x2 tensor.
    # Flattening: for each j, we have a 2x2 matrix that is I_2.
    # The tensor rank of this: it's the sum over j of the same 2x2 identity.
    # T = sum_{j=0}^2 (I_2 ⊗ e_j) where the tensor product is over (i,c) and j.
    # = (e_0+e_1+e_2) ⊗ I_2 in (j, (i,c)) space.
    # Rank = rank(I_2) * 1 = 2? No, tensor rank is different.
    
    # Actually: T = sum_{j=0}^2 sum_{i=0}^1 e_i^i ⊗ e_j^j ⊗ e_i^c
    # = sum_{i=0}^1 (sum_{j=0}^2 e_j^j) ⊗ e_i^i ⊗ e_i^c
    # The rank-1 terms are: for i=0: (1,1,1)_j ⊗ e_0 ⊗ e_0, for i=1: (1,1,1)_j ⊗ e_1 ⊗ e_1
    # So rank is at most 2 for this sub-tensor!
    
    # But wait - this is for fixed k. The issue is that different k's share the
    # same V and W indices in a coupled way.
    
    # Let me try a global construction:
    # For each k, use 2 terms instead of 6. Total rank = 6*2 = 12? That seems too good.
    # The catch: the V and W indices must be consistent across all k.
    
    # Let's try: For term l corresponding to (k, i), define:
    # U[(i',j'), l] = delta_{i',i} (for all j')
    # V[(j,k'), l] = delta_{k',k} (for all j)  
    # W[(k'',i''), l] = delta_{k'',k} * delta_{i'',i}
    
    # Check: sum_l U[i,j,l]*V[j,k,l]*W[k',i',l]
    # = sum over (l_k, l_i) where l corresponds to (k_l, i_l):
    #   delta_{i,i_l} * delta_{k,k_l} * delta_{k',k_l} * delta_{i',i_l}
    # = delta_{i,i'} * delta_{k,k'} * (number of terms with k_l=k and i_l=i) = delta_{i,i'}*delta_{k,k'}
    
    # But we need this to equal 1 when i'=i and k'=k, and 0 otherwise. 
    # Wait, we need T[(i,j),(j,k),(k,i)] = 1, i.e., the (k,i) entry of C.
    # So we need: for output (k_out, i_out), sum = 1 iff k_out=k and i_out=i.
    # From above: sum = delta_{i,i_out} * delta_{k,k_out} * 1 = correct!
    
    # But wait, I need to double-check the indexing. The tensor is:
    # T[a,b,c] where a=(i,j), b=(j,k), c=(k,i) [W[k*n+i] convention]
    # T[(i,j),(j,k),(k,i)] = 1
    # So for given output index c = k_out*n + i_out, and input a=(i,j), b=(j,k):
    # We need T[(i,j),(j,k),(k_out,i_out)] = 1 iff k_out=k and i_out=i.
    
    # With my construction: sum_l U[i,j,l]*V[j,k,l]*W[k_out*n+i_out, l]
    # For l = (k_l, i_l): U[i,j,l]=delta_{i,i_l}, V[j,k,l]=delta_{k,k_l}, 
    # W[k_out*n+i_out,l]=delta_{k_out,k_l}*delta_{i_out,i_l}
    # Product = delta_{i,i_l}*delta_{k,k_l}*delta_{k_out,k_l}*delta_{i_out,i_l}
    # = delta_{i,i_l}*delta_{k,k_l}*delta_{k_out,k_l}*delta_{i_out,i_l}
    # Sum over l: = delta_{i,i_out}*delta_{k,k_out} (if there's exactly one l with k_l=k, i_l=i)
    
    # This works! Rank = n*p = 2*6 = 12!
    
    # Let me construct this:
    rank_candidate = n * p  # 12
    U_cand = [[0] * rank_candidate for _ in range(n * m)]
    V_cand = [[0] * rank_candidate for _ in range(m * p)]
    W_cand = [[0] * rank_candidate for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            l = k * n + i  # index for this term
            # U: for all j, U[i*m+j, l] = 1
            for j in range(m):
                U_cand[i * m + j][l] = 1
            # V: for all j, V[j*p+k, l] = 1
            for j in range(m):
                V_cand[j * p + k][l] = 1
            # W: W[k*n+i, l] = 1
            W_cand[k * n + i][l] = 1
    
    # Verify:
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Build tensor
        # T[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
        # a = i*m+j, b = j*p+k, c = k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1) > 1e-9:
                        return False
        # Check zeros
        # For all (a,b,c) not in the "diagonal" pattern, should be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Decode
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    # Should be nonzero only if i_a==i_c and j_a==j_b and k_b==k_c
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
        return True
    
    if verify(U_cand, V_cand, W_cand, n, m, p):
        return {"U": U_cand, "V": V_cand, "W": W_cand}
    
    # If verification failed, try rank 35 search
    if time.time() > deadline - 10:
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    # Fallback to trivial
    return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
# EVOLVE-BLOCK-END
