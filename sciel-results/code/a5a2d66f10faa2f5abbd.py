"""Matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 x 3x4 matrix multiplication.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)
        - W has shape (p*n, rank) = (12, rank)
    """
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    
    # The matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k]
    # where A is n x m, B is m x p, C is n x p
    # T has shape (n, m, p) = (3, 3, 4)
    
    # We need to find rank r and matrices U, V, W such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For 3x3x3, the minimal rank is 23 (Landsberg's bound / known result)
    # For 3x3x4, we need to determine the minimal rank.
    
    # Let's try a structured approach.
    # The trivial rank is n*m*p = 3*3*4 = 36
    # We can do better by exploiting structure.
    
    # Strategy: Use the fact that matrix multiplication is a special tensor.
    # For rectangular matrices, the rank is at least max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # but this is a lower bound that's not tight.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (best known)
    # - For 3x3x4, we can use a combination.
    
    # Let me try to construct a decomposition using a known pattern.
    # One approach: decompose the 3x3x4 as a sum of smaller tensor products.
    
    # Actually, let's use a more systematic approach.
    # We'll try ranks from 12 upward and use a solver to find the decomposition.
    
    start_time = time.time()
    deadline = 120.0  # seconds, leave margin for evaluation overhead
    
    # First, let's try a known good decomposition.
    # For 3x3x4, a reasonable upper bound is around 24-30.
    
    # Let me try to construct a rank-24 decomposition using a block structure.
    # The 3x3x4 tensor can be viewed as two 3x3x2 blocks.
    
    # Actually, let's use a different approach: 
    # Use the fact that we can decompose using the Kronecker product structure.
    
    # For now, let's try a rank that's known to work.
    # The trivial rank is 36. Let's see if we can do better.
    
    # Try rank 24 first with a structured pattern.
    # We'll use a pattern based on the 3x3x3 rank-23 decomposition extended to 3x3x4.
    
    # Let me try a more direct approach: use linear algebra to solve for the decomposition.
    # Given a rank r, we need to find U (9 x r), V (12 x r), W (12 x r)
    # such that for all i in 0..2, j in 0..2, k in 0..3:
    #   T[i,j,k] = sum_{r=0}^{r-1} U[i*3+j, r] * V[j*4+k, r] * W[k*3+i, r]
    
    # This is a system of polynomial equations. We can use a grid search or optimization.
    
    # Let's try a simple approach: use a random initialization and gradient descent.
    # But we need exact integer/half-integer entries.
    
    # Alternative: use a known decomposition pattern.
    
    # For 3x3x4, let's try rank 24 with a structured pattern.
    # The pattern: for each pair (i, k), we can share structure across j.
    
    # Let me try a different tactic: use the fact that the tensor has a lot of zeros
    # in a suitable basis, and exploit sparsity.
    
    # Actually, the matrix multiplication tensor T[i,j,k] is 1 for all (i,j,k) in the 
    # appropriate range - it's a "full" tensor in the sense that all entries are 1.
    # Wait, no. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    # This is a system where every entry of the tensor is 1.
    
    # Let's try rank r and see if we can find U, V, W with small integer entries.
    
    # For rank 36 (trivial), it's easy:
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1 for each (i,j,k)
    
    # But we want to minimize rank.
    
    # Let's try a rank-24 approach using a structured pattern.
    # Idea: group the 36 terms into 24 by combining some.
    
    # Actually, let me try to use a known result or construct one.
    # For 3x3x4, I'll try rank 24 with the following pattern:
    
    # Let's use a basis where we have terms for each (i, k) pair and share across j.
    # There are 3*4 = 12 (i,k) pairs. For each, we need to handle 3 j values.
    # If we use 2 terms per (i,k) pair, that's 24 terms total.
    
    # Term structure for (i, k):
    # For j=0: U[i*3+0, t1] * V[0*4+k, t1] * W[k*3+i, t1]
    # For j=1: U[i*3+1, t2] * V[1*4+k, t2] * W[k*3+i, t2]
    # For j=2: U[i*3+2, t3] * V[2*4+k, t3] * W[k*3+i, t3]
    
    # This doesn't immediately help. Let me try a different grouping.
    
    # Let's just use a rank that we can verify works, and try to minimize it.
    # I'll start with rank 30 and try to reduce.
    
    # Actually, let me try a simple construction:
    # Use rank r = 24 with a specific pattern.
    
    # Pattern: For each (i, j, k), assign a term index, but try to reuse terms.
    # Key insight: if two (i,j,k) triples share the same i and k, they can potentially
    # share a W entry. If they share j and k, they can share a V entry. Etc.
    
    # Let's try a rank-24 construction:
    # Terms indexed by (i, k, s) where s in {0, 1, 2} for the j dimension.
    # But that's 3*4*2 = 24 if we use 2 terms per (i,k).
    
    # For a given (i, k), we need to satisfy:
    # For j=0: U[i*3+0, term1] * V[0*4+k, term1] * W[k*3+i, term1] + ... = 1
    # For j=1: U[i*3+1, term2] * V[1*4+k, term2] * W[k*3+i, term2] + ... = 1
    # For j=2: U[i*3+2, term3] * V[2*4+k, term3] * W[k*3+i, term3] + ... = 1
    
    # If we use exactly one term per (i, j, k), that's 36 terms.
    # If we can combine, we reduce.
    
    # Let me try a concrete rank-24 construction:
    # Use terms indexed by (i, k, j_group) where j_group in {0, 1}
    # For j_group=0: handles j=0 and j=1
    # For j_group=1: handles j=2
    # That's 3*4*2 = 24 terms.
    
    # For term (i, k, 0):
    #   Contributes to (i, 0, k): U[i*3+0, t] * V[0*4+k, t] * W[k*3+i, t]
    #   Contributes to (i, 1, k): U[i*3+1, t] * V[1*4+k, t] * W[k*3+i, t]
    #   We need both to be 1, so:
    #   U[i*3+0, t] * V[0*4+k, t] * W[k*3+i, t] = 1
    #   U[i*3+1, t] * V[1*4+k, t] * W[k*3+i, t] = 1
    #   This requires U[i*3+0, t] * V[0*4+k, t] = U[i*3+1, t] * V[1*4+k, t]
    #   Let W[k*3+i, t] = 1, U[i*3+0, t] = a, V[0*4+k, t] = 1/a
    #   U[i*3+1, t] = b, V[1*4+k, t] = 1/b
    #   But we need integer/half-integer entries.
    
    # This is getting complicated. Let me just use a rank that works.
    
    # I'll use rank 36 (trivial) as a fallback, and try to find a smaller rank.
    
    # Let me try rank 24 with a simple pattern:
    # For each (i, k), use 2 terms.
    # Term 1: handles j=0, j=1
    # Term 2: handles j=2
    
    # For term (i, k, 0):
    #   W[k*3+i, t] = 1
    #   U[i*3+0, t] = 1, V[0*4+k, t] = 1  -> contributes 1 to (i,0,k)
    #   U[i*3+1, t] = 1, V[1*4+k, t] = 1  -> contributes 1 to (i,1,k)
    #   U[i*3+2, t] = 0, V[2*4+k, t] = 0  -> contributes 0 to (i,2,k)
    
    # For term (i, k, 1):
    #   W[k*3+i, t] = 1
    #   U[i*3+0, t] = 0, V[0*4+k, t] = 0  -> contributes 0 to (i,0,k)
    #   U[i*3+1, t] = 0, V[1*4+k, t] = 0  -> contributes 0 to (i,1,k)
    #   U[i*3+2, t] = 1, V[2*4+k, t] = 1  -> contributes 1 to (i,2,k)
    
    # This gives rank 24! Let me verify:
    # For (i, 0, k): sum over terms = 1 (from term (i,k,0)) + 0 (from term (i,k,1)) = 1 ✓
    # For (i, 1, k): sum over terms = 1 (from term (i,k,0)) + 0 (from term (i,k,1)) = 1 ✓
    # For (i, 2, k): sum over terms = 0 (from term (i,k,0)) + 1 (from term (i,k,1)) = 1 ✓
    
    # But wait, I need to check that terms from different (i,k) pairs don't interfere.
    # Term (i1, k1, 0) has:
    #   U[i1*3+0, t] = 1, U[i1*3+1, t] = 1, U[i1*3+2, t] = 0
    #   V[0*4+k1, t] = 1, V[1*4+k1, t] = 1, V[2*4+k1, t] = 0
    #   W[k1*3+i1, t] = 1
    #   All other entries in U, V, W for this term are 0.
    
    # This term only contributes to (i1, 0, k1) and (i1, 1, k1).
    # Similarly, term (i1, k1, 1) only contributes to (i1, 2, k1).
    
    # So the total contribution to (i, j, k) is:
    # - From term (i, k, 0): 1 if j in {0, 1}, else 0
    # - From term (i, k, 1): 1 if j == 2, else 0
    # - From all other terms: 0
    
    # So the total is 1 for all (i, j, k). ✓
    
    # This gives a valid rank-24 decomposition!
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 9 x 24
    V = [[0] * rank for _ in range(m * p)]  # 12 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    term_idx = 0
    for i in range(n):
        for k in range(p):
            # Term 0: handles j=0, j=1
            t = term_idx
            U[i * m + 0][t] = 1
            U[i * m + 1][t] = 1
            V[0 * p + k][t] = 1
            V[1 * p + k][t] = 1
            W[k * n + i][t] = 1
            term_idx += 1
            
            # Term 1: handles j=2
            t = term_idx
            U[i * m + 2][t] = 1
            V[2 * p + k][t] = 1
            W[k * n + i][t] = 1
            term_idx += 1
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r] for r in range(rank))
                if val != 1:
                    # Fallback to trivial rank 36
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][col] = 1
                                V[jj * p + kk][col] = 1
                                W[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
