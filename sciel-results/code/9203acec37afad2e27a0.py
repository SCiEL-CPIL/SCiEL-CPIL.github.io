"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    dim_U = n * m  # 6
    dim_V = m * p  # 18
    dim_W = p * n  # 12
    
    # The tensor T[A_idx, B_idx, C_idx] = 1 if consistent, 0 otherwise
    # A_idx = i*m+j, B_idx = j'*p+k, C_idx = k'*n+i'
    # T = 1 iff i==i', j==j', k==k'
    
    # Build the full tensor as a 3D array for verification
    # But for rank computation, we can use unfoldings
    
    # Strategy: try ranks from a lower bound upward, and for each rank,
    # attempt to find a decomposition by solving a nonlinear system.
    # For efficiency, we'll use a constructive approach.
    
    # Lower bound: rank >= max(dim_U, dim_V, dim_W) / min(dim_U, dim_V, dim_W) ... not quite.
    # Actually, rank >= max over all unfoldings of the matrix rank.
    
    # Let's compute the rank of the (dim_U x dim_V) unfolding by fixing each C_idx
    # T_C[C_idx, A_idx, B_idx] - for each C_idx, this is a matrix of size dim_U x dim_V
    # The tensor rank is at least the max rank of these slices.
    
    # For C_idx = k'*n+i', the slice is:
    # M[i*m+j, j'*p+k] = 1 if i==i', j==j', k==k' else 0
    # This is an outer product of vectors: for fixed (i', k'), 
    # M = u_{i'} * v_{k'}^T where u_{i'} has 1s at positions i'*m+j for j in range(m)
    # and v_{k'} has 1s at positions j'*p+k' for j' in range(m)
    
    # So each slice has rank 1! The max slice rank is 1.
    # But tensor rank can be higher than max slice rank.
    
    # Let's try to construct a low-rank decomposition directly.
    # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Try rank = m*p = 18:
    # For each (j, k) pair, create terms that handle all i values.
    # Use n terms per (j,k)? That's 36 again.
    
    # Alternative: use the fact that for fixed (j,k), we need to produce delta_{i,i'} 
    # which is an n x n identity. We can decompose identity_n into n rank-1 terms.
    
    # But we can share the V factor across i values.
    # For fixed (j,k), let V[j*p+k, c] = 1 for a set of c values.
    # Then U[i*m+j, c] * W[k*n+i', c] must sum to delta_{i,i'}.
    
    # This is equivalent to: for each (j,k), we need a decomposition of I_n into 
    # rank-1 matrices using shared structure.
    
    # I_n can be written as sum of n rank-1 terms (the standard basis).
    # So for each (j,k), we need n terms. Total: m*p*n = 36.
    
    # To do better, we need to share terms across different (j,k) pairs.
    
    # Idea: Use the same set of c indices for all (j,k) pairs, but with different 
    # values in U and W.
    
    # Let's try rank = n*p = 12:
    # For each (i', k') pair (which is C_idx), we want to isolate it.
    # Hmm, this is tricky.
    
    # Let me try a different approach: use random initialization and gradient descent
    # to minimize the error, then round to rationals.
    
    # Actually, let's try a structured construction.
    # 
    # Observation: The tensor is T[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # This is the tensor product of three identity tensors!
    # T = I_n ⊗ I_m ⊗ I_p (as a tensor product)
    # 
    # The rank of a tensor product is the product of ranks:
    # rank(T) = rank(I_n) * rank(I_m) * rank(I_p) = n * m * p = 36
    # 
    # Wait, is that right? The tensor product of decompositions:
    # If A = sum_{a=1}^{r_A} a^{(a)}_1 ⊗ a^{(a)}_2 ⊗ ... 
    # and B = sum_{b=1}^{r_B} b^{(b)}_1 ⊗ b^{(b)}_2 ⊗ ...
    # then A ⊗ B = sum_{a,b} (a_1⊗b_1) ⊗ (a_2⊗b_2) ⊗ ...
    # So rank(A⊗B) <= rank(A) * rank(B)
    # 
    # And I_n has rank n (as a tensor? No, I_n as a matrix has rank n, but as a 
    # 2-tensor it's just a matrix).
    # 
    # Hmm, I_n is a 2-tensor (matrix), not a 3-tensor. The tensor T is a 3-tensor
    # with T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'} but wait, that's not right.
    
    # Let me reclarify. The tensor has three modes:
    # Mode 1: index a = i*m+j (size n*m = 6)
    # Mode 2: index b = j'*p+k (size m*p = 18)
    # Mode 3: index c = k'*n+i' (size n*p = 12)
    # 
    # T[a, b, c] = 1 iff i==i', j==j', k==k' where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # 
    # This is NOT a simple tensor product of identities.
    
    # Let me just go with the schoolbook decomposition of rank 36, which is guaranteed correct.
    # But first, let me try to verify if a lower rank is possible by checking small cases.
    
    # For (1,1,1): rank 1
    # For (2,1,1): T[i,j,k;i',j',k'] with j=j'=0, k=k'=0
    # T[i,0,0; i',0,0] = delta_{i,i'}
    # This is I_2, which has rank 2 as a matrix, so tensor rank >= 2.
    # Indeed, rank = 2.
    
    # For (2,2,1): 
    # T[i,j,0; i',j',0] = delta_{i,i'}delta_{j,j'}
    # This is I_2 ⊗ I_2 = I_4, rank 4.
    
    # So for (n,m,1): rank = n*m
    # For (2,3,6): by similar logic, rank = n*m*p = 36?
    
    # But wait, that's only if the tensor is a pure tensor product. Let me verify 
    # with a smaller case: (2,2,2).
    # T[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Is this equal to I_2 ⊗ I_2 ⊗ I_2?
    # (I_2 ⊗ I_2 ⊗ I_2)[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Yes! So T = I_n ⊗ I_m ⊗ I_p as a 3-tensor with modes (n*m, m*p, n*p)?
    
    # No wait, the modes don't match. I_n ⊗ I_m ⊗ I_p has modes (n, m, p) x (n, m, p) x (n, m, p)
    # or something like that. The tensor T has modes of sizes n*m, m*p, n*p.
    
    # I think the correct statement is:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # 
    # This can be seen as: fix the "matching" pattern. For each (i,j,k) triple,
    # T has a 1 at position (i*m+j, j*p+k, k*n+i) and 0 elsewhere.
    # 
    # So T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # 
    # This is a sum of n*m*p = 36 rank-1 tensors, so rank <= 36.
    # 
    # Can we do better? The question is whether these 36 rank-1 tensors are linearly 
    # independent in the tensor space.
    
    # Let's check linear independence. Suppose sum_{i,j,k} c_{i,j,k} * e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} = 0
    # This means for all (a,b,c): sum_{i,j,k: i*m+j=a, j*p+k=b, k*n+i=c} c_{i,j,k} = 0
    # 
    # For a given (a,b,c), there is at most one (i,j,k) that satisfies the constraints:
    # a = i*m+j => i = a//m, j = a%m
    # b = j'*p+k => j' = b//p, k = b%p
    # c = k'*n+i' => k' = c//n, i' = c%n
    # 
    # For consistency: j == j', k == k', i == i'
    # So a%m == b//p, b%p == c//n, a//m == c%n
    # 
    # If these conditions hold, there's exactly one (i,j,k) contributing, so c_{i,j,k} = 0.
    # If not, no (i,j,k) contributes, so the equation is 0=0 (no constraint).
    # 
    # Therefore, the 36 rank-1 tensors are linearly independent, and rank = 36.
    
    # So the minimal rank is exactly 36! The schoolbook decomposition is optimal.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
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
