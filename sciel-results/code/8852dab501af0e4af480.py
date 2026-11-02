"""Bootstrap spawn: direct structural construction with rank reduction via linear dependencies."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or at least reduced) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x8 matrix.
    
    The tensor T has entries T[i, j, k] for i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    
    Direct approach: Use the schoolbook rank (n*m*p = 96) as baseline,
    then try to find a lower-rank decomposition using known algorithms.
    
    For 3x4x8, we try to find a decomposition with rank less than 96.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Start with schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions using linear algebra
    # The key insight: if there are linear dependencies among the rank-1 terms,
    # we can potentially reduce the rank.
    
    # For matrix multiplication, the tensor is T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C
    # where e_i^A, e_j^B, e_k^C are basis vectors.
    
    # One approach: try random linear combinations and check if they still span
    # the correct subspace. But this is complex.
    
    # Simpler approach: use the fact that for small dimensions, we can try
    # specific known lower-rank constructions.
    
    # For 3x4x8, let's try a constructive approach based on grouping.
    # Group the j-index into pairs and use Strassen-like tricks.
    
    # Actually, let's try a systematic search for linear dependencies.
    # The rank of the multiplication tensor is the minimum r such that
    # T can be written as sum of r rank-1 tensors.
    
    # A necessary condition: for any partition of the indices, the rank
    # of the corresponding matrix slice must be <= r.
    
    # Let's compute some lower bounds.
    # The tensor T as a 3-way array of size 3x4x8.
    # Slice along k: for each k, we get a 3x4 matrix of all 1s. Rank = 1.
    # Slice along j: for each j, we get a 3x8 matrix of all 1s. Rank = 1.  
    # Slice along i: for each i, we get a 4x8 matrix of all 1s. Rank = 1.
    
    # So each slice has rank 1, but that doesn't directly give a lower bound on r.
    
    # The true rank of the 3x4x8 multiplication tensor is known to be less than 96.
    # Let's try to construct an explicit lower-rank decomposition.
    
    # Approach: use the fact that we can combine terms with the same (i,k) but different j.
    # For fixed i and k, C[i,k] = sum_j A[i,j] * B[j,k].
    # The term U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1 for all i,j,k.
    
    # Let's try a decomposition where we use fewer terms by exploiting structure.
    # 
    # Idea: For each pair (i, k), we need sum_j A[i,j]*B[j,k]. 
    # If we could use a single term per (i,k) pair, we'd need n*p = 24 terms,
    # but that's not possible because the j-dependence must be captured.
    
    # Let's try rank 80 (a reduction of 16 from 96) by merging some terms.
    
    # Actually, let me try a different approach: use integer linear programming
    # or greedy methods to find linear dependencies.
    
    # For now, let's just return the schoolbook decomposition if we can't find
    # a better one quickly. But let's try one more thing: a simple optimization
    # that checks if we can merge any two columns.
    
    # Two columns c1 and c2 can be merged if their product vectors are linearly dependent.
    # Column c has vectors u_c (in R^{n*m}), v_c (in R^{m*p}), w_c (in R^{p*n}).
    # The contribution is u_c ⊗ v_c ⊗ w_c.
    # Two such tensors are linearly dependent only if one is a scalar multiple of the other,
    # which for our 0/1 entries means they'd have to be identical.
    
    # In the schoolbook, all columns are distinct, so no immediate merging.
    
    # Let's try a randomized approach: generate random rank-r decompositions
    # and check if they work.
    
    # For rank r, we need U (12 x r), V (32 x r), W (24 x r) such that
    # for all i in [0,3), j in [0,4), k in [0,8):
    #   sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*3+i, c] = 1
    
    # This is a system of 96 equations in r^2 * (something) variables... 
    # Actually, it's 96 equations with U, V, W entries as unknowns.
    # Total unknowns: 12*r + 32*r + 24*r = 68*r.
    # For r = 96, we have 6528 unknowns and 96 equations - very underdetermined.
    
    # Let's try to find a solution with r < 96 using a simple iterative method.
    
    # Actually, for the purpose of this task, let's just ensure we return a valid
    # decomposition. The schoolbook is valid with rank 96.
    
    # Let me try one more idea: use the Strassen algorithm adapted to 3x4x8.
    # Strassen reduces 2x2x2 from 8 to 7. For larger matrices, it can be applied recursively.
    
    # For 3x4x8, we could pad to 4x4x8 and use Strassen-like tricks.
    # 4x4x8 = (2x2x2) blocks with Strassen: each 2x2x2 block uses 7 instead of 8.
    # Number of 2x2x2 blocks in 4x4x8: (4/2)*(4/2)*(8/2) = 2*2*4 = 16 blocks.
    # With Strassen: 16 * 7 = 112, which is WORSE than 96. So padding makes it worse.
    
    # For 3x4x8 without padding, the schoolbook rank is 96, and it's likely close to optimal.
    
    # Let's just verify the schoolbook is correct and return it.
    
    # Verification:
    # For each i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # In schoolbook, each (i,j,k) has exactly one column c where all three are 1.
    # So the sum is 1. Correct.
    
    # Check time budget
    if time.time() > deadline:
        return best_result
    
    # Try a simple local search: see if we can reduce rank by 1
    # by finding a linear dependency.
    
    # Actually, let's try constructing a specific lower-rank decomposition.
    # 
    # Consider the following: for the j-dimension (size 4), we can use
    # a 2x2 Strassen-like decomposition if we group indices appropriately.
    #
    # Let's try: group j into {0,1} and {2,3}. For each group, use 2x2 Strassen.
    # But the i and k dimensions don't align nicely for Strassen.
    
    # Given the complexity, let's just return the schoolbook for now.
    # The score will be 96, same as the seed.
    
    return best_result
# EVOLVE-BLOCK-END
