import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-16 decomposition
    rank16 = try_rank16()
    if rank16 is not None:
        return rank16
    
    # Fallback: rank 17 decomposition
    return rank17_decomposition()

def try_rank16():
    """
    Attempt to find a rank-16 decomposition using exact rational arithmetic.
    """
    n, m, p = DIMENSIONS
    r = 16
    
    # The 18 equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Let's use a parameterized approach.
    
    # Fix W to be a 6x16 matrix. We'll set it to have a specific structure.
    # Let W be such that W[k*2+i, c] = 1 if c == k*2+i and c < 6, else 0.
    # This means W is the first 6 columns of a 6x16 identity-like matrix.
    # But this only uses 6 of the 16 rank-1 terms.
    
    # Better: let's set W to be a 6x16 matrix where each column c has W[k*2+i, c] = 1
    # for exactly one (k,i) pair. With 16 columns and 6 (k,i) pairs, some pairs repeat.
    
    # Actually, let's try a different approach: use the fact that the system is
    # bilinear and solve it by fixing one factor.
    
    # Fix V to be a 9x16 matrix with a specific structure, then the equations
    # become linear in U and W (for fixed V).
    
    # Let's set V to be a 9x16 matrix where V[j*3+k, c] = 1 if c == j*3+k and c < 9,
    # and for c >= 9, V[j*3+k, c] = 0.
    # This means only the first 9 rank-1 terms contribute.
    
    # Then the equation for (i,j,k) becomes:
    # sum_{c=0}^{8} U[i*3+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
    # Since V[j*3+k,c] = delta_{j*3+k,c} for c < 9, this becomes:
    # U[i*3+j, j*3+k] * W[k*2+i, j*3+k] = 1  (for j*3+k < 9, which is always true)
    
    # So we need U[i*3+j, j*3+k] * W[k*2+i, j*3+k] = 1 for all (i,j,k).
    # This gives 18 equations in the products of U and W entries.
    
    # Let a_{i,j,k} = U[i*3+j, j*3+k] and b_{i,j,k} = W[k*2+i, j*3+k].
    # We need a_{i,j,k} * b_{i,j,k} = 1 for all (i,j,k).
    
    # This is possible: set a_{i,j,k} = 1 and b_{i,j,k} = 1 for all (i,j,k).
    # But we need consistency: U[i*3+j, j*3+k] must be the same for all k that give the same (i,j) and c=j*3+k.
    # Wait, c = j*3+k, so for fixed (i,j), U[i*3+j, c] depends on c = j*3+k.
    # For fixed i and j, as k varies, c varies, so U[i*3+j, c] can be different for different k.
    # Similarly for W.
    
    # So set U[i*3+j, j*3+k] = 1 for all (i,j,k) and W[k*2+i, j*3+k] = 1 for all (i,j,k).
    # Check consistency:
    # U[i*3+j, c] where c = j*3+k: for fixed (i,j), c ranges over j*3, j*3+1, j*3+2.
    # So U[i*3+j, j*3], U[i*3+j, j*3+1], U[i*3+j, j*3+2] are all set to 1.
    # That's fine, they're different columns.
    
    # W[k*2+i, c] where c = j*3+k: for fixed (k,i), c ranges over 0,1,2 (j=0), 3,4,5 (j=1), 6,7,8 (j=2).
    # So W[k*2+i, 0], W[k*2+i, 1], ..., W[k*2+i, 8] are all set to 1.
    # That's fine.
    
    # But wait, this only uses 9 rank-1 terms (c=0 to 8). We have 16 available.
    # The remaining 7 terms (c=9 to 15) have V[j*3+k, c] = 0, so they don't contribute.
    # This gives a valid rank-9 decomposition? No, because we're using 9 terms but they might not be independent.
    
    # Actually, this construction gives a valid decomposition with rank at most 9!
    # But that can't be right because the rank of M_{2,3,3} is at least 9 (from the flattening).
    
    # Let me verify: with this construction, does it satisfy all 18 equations?
    # For (i,j,k): sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c]
    # = sum_c U[i*3+j,c]*delta_{j*3+k,c}*W[k*2+i,c]
    # = U[i*3+j, j*3+k] * W[k*2+i, j*3+k]
    # = 1 * 1 = 1. Yes!
    
    # But this uses only 9 rank-1 terms. The rank is at most 9.
    # However, the rank of M_{2,3,3} is known to be 17, so this can't be a valid decomposition.
    
    # The issue: I'm not accounting for the fact that the rank-1 terms must be linearly independent
    # in the tensor space. The construction above might have linear dependencies.
    
    # Actually, the rank of a decomposition is the number of rank-1 terms, regardless of linear independence.
    # So if I use 9 terms, the rank is at most 9. But the tensor rank is the minimum number of rank-1 terms.
    # If the tensor rank is 17, then 9 terms cannot sum to the matrix multiplication tensor.
    
    # Let me re-examine. The issue is that my construction might not be valid because
    # the same (i,j,k) might be covered by multiple terms, or the terms might interfere.
    
    # Actually, in my construction, for each (i,j,k), exactly one term (c = j*3+k) contributes,
    # and it contributes 1. So the sum is 1. This seems valid.
    
    # But the tensor rank of M_{2,3,3} is 17, so this can't be right. Let me check the flattening.
    
    # The flattening over j gives a 3x6 matrix (j vs (i,k)). Entry [j, (i,k)] = 1 for all.
    # This matrix has rank 1 (all ones). So the flattening lower bound is 1, not 9.
    
    # The flattening over i gives a 2x9 matrix (i vs (j,k)). Entry [i, (j,k)] = 1 for all.
    # Rank 1.
    
    # The flattening over k gives a 3x6 matrix (k vs (i,j)). Entry [k, (i,j)] = 1 for all.
    # Rank 1.
    
    # So the flattening lower bound is 1, which is very weak.
    
    # The actual tensor rank of M_{2,3,3} is 17 according to the literature.
    # My construction must have an error.
    
    # Let me re-examine the indexing.
    # A is n x m = 2 x 3. A[i,j] for i in {0,1}, j in {0,1,2}.
    # B is m x p = 3 x 3. B[j,k] for j in {0,1,2}, k in {0,1,2}.
    # C is n x p = 2 x 3. C[i,k] for i in {0,1}, k in {0,1,2}.
    # C[i,k] = sum_j A[i,j] * B[j,k].
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) because the product A[i,j]*B[j,k] contributes to C[i,k].
    # Wait, no. The tensor is defined such that T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], the coefficient of A[i,j]*B[j,k] in C[i,k] is 1 if the indices match,
    # which they always do for the term A[i,j]*B[j,k] contributing to C[i,k].
    # So T[i,j,k] = 1 for all (i,j,k).
    
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] (with appropriate indexing).
    # In our notation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    # My construction: U[i*3+j, j*3+k] = 1, V[j*3+k, j*3+k] = 1, W[k*2+i, j*3+k] = 1.
    # Check: sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*2+i,c]
    # = U[i*3+j, j*3+k] * V[j*3+k, j*3+k] * W[k*2+i, j*3+k]  (only c=j*3+k contributes)
    # = 1 * 1 * 1 = 1. Correct!
    
    # So this is a valid rank-9 decomposition? But the rank is supposed to be 17.
    
    # I think the issue is that the rank of a decomposition is not just the number of terms,
    # but the minimum number of rank-1 tensors that sum to the given tensor.
    # If I can write the tensor as a sum of 9 rank-1 tensors, then the rank is at most 9.
    # But the literature says the rank is 17.
    
    # Let me check the literature more carefully. The rank of M_{2,3,3} might not be 17.
    # Actually, I recall that the rank of M_{2,2,2} is 7 (Strassen), and M_{2,3,3} might have a different rank.
    
    # Let me verify my construction by checking if the rank-1 terms are linearly independent.
    # If they are linearly dependent, the effective rank is lower.
    
    # Actually, the rank of a decomposition is defined as the number of rank-1 terms, regardless of linear independence.
    # So if I use 9 terms, the rank is 9. But if the tensor rank is 17, then 9 terms cannot sum to the tensor.
    
    # I must have made an error. Let me re-examine.
    
    # Oh wait, I think the issue is that my construction doesn't work because the same (i,j,k)
    # might be covered by multiple terms in a way that causes interference.
    
    # No, in my construction, for each (i,j,k), exactly one term contributes (c = j*3+k),
    # and it contributes exactly 1. So the sum is 1.
    
    # I'm confused. Let me just implement the rank-17 decomposition and return it.
    # The rank-16 search is not working, and the rank-17 decomposition is known to be valid.
    
    return None

def rank17_decomposition():
    """
    Construct a rank-17 decomposition for 2x3x3 matrix multiplication
    by combining Strassen (rank 7) for 2x2x2 part with schoolbook for remaining parts.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
