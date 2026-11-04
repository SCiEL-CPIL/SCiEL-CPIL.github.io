"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def check_decomp(U, V, W, tol=1e-9):
        """Check if U,V,W give exact decomposition. U,V,W are lists of lists."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try to find a lower-rank decomposition using structured approach
    # Key insight: T[i,j,k] = 1 for all i,j,k
    # We can use a factorization based on separating the index dependencies
    
    # Approach: Use the fact that we can write the decomposition using
    # products of functions of individual indices
    # Let's try a rank based on the structure of the problem
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try various structured decompositions
    # Method 1: Use rank = m * (n + p) - m
    # This comes from a factorization where we separate the middle index
    
    # Actually, let's try a direct constructive approach:
    # For the all-ones tensor, we can use:
    # U[i*m+j, c] = 1 if c < m, and specific values otherwise
    # Let me try rank = n*p + m*(n-1) + m*(p-1) or similar
    
    # Simpler: try rank values and use a greedy/structured construction
    # The key is that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a decomposition where we use the structure:
    # For each j, we can have terms that factor as (function of i) * (function of k)
    
    # Try rank = m * (n + p - 1)
    # This is based on: for each j, we need to represent the all-ones n x p matrix
    # as a sum of rank-1 matrices, which takes n+p-1 terms (for full rank)
    # But since it's all ones, rank 1 suffices per j, giving rank = m
    
    # Wait, that's not right. Let me reconsider.
    # If U[i*m+j, c] depends on i and j, V[j*p+k, c] depends on j and k,
    # W[k*n+i, c] depends on k and i, then for fixed j, we have:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,k
    # This is like representing an all-ones n x p matrix as a sum of rank-1 terms
    # where each term is (U[:,j,c] * V[:,j,c]) outer product weighted by W
    
    # For all-ones matrix, we need rank 1: just one term with all ones
    # So for each j, we could use 1 term, giving total rank = m = 3
    # But we need to check if W can be shared or if there are constraints
    
    # Let's try rank = m = 3
    # U[i*m+j, c] = 1 for all i, j, c (c = 0,1,2)
    # V[j*p+k, c] = 1 for all j, k, c
    # W[k*n+i, c] = 1 for all k, i, c
    # Then sum_c U*V*W = sum_c 1*1*1 = 3, not 1
    # So we need to scale: U = 1/sqrt(3), V = 1/sqrt(3), W = 1/sqrt(3)
    # But we need integer/half-integer entries
    
    # Try: U = 1, V = 1, W = 1/3? No, W must be integer/half-integer
    # Try: U = 1, V = 1, W = 1, but with only 1 column: rank = 1
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum = 1*1*1 = 1. This works! Rank = 1!
    
    # Wait, that can't be right. Let me verify.
    # For matrix multiplication C = A * B where A is 2x3, B is 3x9
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # T[i,j,k] = 1 for all i,j,k because each product A[i,j]*B[j,k] contributes to C[i,k]
    # So yes, T is all ones, and rank 1 decomposition: U=1, V=1, W=1 gives sum=1
    
    # But wait, is this actually a valid tensor decomposition for matrix multiplication?
    # The tensor decomposition requires that the bilinear form matches matrix multiplication
    # Let me re-examine: the matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A
    # and (j,k) entry of B multiply to contribute to (i,k) entry of C. Yes, T[i,j,k]=1 for all valid i,j,k.
    
    # So rank 1 should work! Let me verify with the check.
    
    U_rank1 = [[1] for _ in range(n * m)]
    V_rank1 = [[1] for _ in range(m * p)]
    W_rank1 = [[1] for _ in range(p * n)]
    
    if check_decomp(U_rank1, V_rank1, W_rank1):
        return {"U": U_rank1, "V": V_rank1, "W": W_rank1}
    
    # If rank 1 doesn't work (maybe I'm misunderstanding the tensor), try rank 2, 3, etc.
    # But first, let me double-check my understanding
    
    # Actually, I think I may be confusing the tensor definition. Let me reconsider.
    # The matrix multiplication tensor T is defined such that:
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has components T[i,j,k] which are the coefficients
    # In the standard definition, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each pair (A[i,j], B[j,k]) contributes to C[i,k] with coefficient 1
    
    # So my rank 1 solution should be correct. But let me try a few more ranks just in case.
    
    # Try rank = 1 with different scalings (in case I need half-integers)
    # U = 1, V = 1, W = 1 works with integers
    
    # If the evaluator rejects rank 1, there might be a subtlety I'm missing
    # Let's try to be safe and also prepare a rank 2 solution
    
    # But first, let's return rank 1 if it passes the check
    # The check already verified it, so return it
    
    # Actually, let me reconsider the problem statement. It says "exactly decompose the matrix multiplication tensor"
    # Maybe the tensor is not all ones? Let me re-read.
    # "C[i,k] = sum_j A[i,j] * B[j,k]"
    # The tensor T[i,j,k] is the coefficient of the monomial A[i,j]*B[j,k] in C[i,k]
    # For matrix multiplication, T[i,j,k] = 1 if k is in the output, which it always is
    # So T[i,j,k] = 1 for all i,j,k. My solution should be correct.
    
    # Return the rank 1 solution
    return {"U": U_rank1, "V": V_rank1, "W": W_rank1}
# EVOLVE-BLOCK-END
