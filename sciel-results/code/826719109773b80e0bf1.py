"""Matrix multiplication 3x5 x 5x5 rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (standard matrix mult tensor)
    """
    n, m, p = N, M, P
    
    # The standard matrix multiplication tensor T[i,j,k] = 1 if the indices form a valid
    # multiplication path. Specifically, for matrix C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 for all valid (i,j,k).
    
    # Rank of this tensor: we need to find the minimum number of rank-1 tensors
    # that sum to T. A rank-1 tensor is u_i * v_j * w_k.
    
    # The trivial (schoolbook) decomposition has rank n*m*p = 75.
    # We need to find if we can do better.
    
    # Known results: For matrix multiplication, the rank equals the tensor rank.
    # For small matrices, we can try to find lower ranks via integer/half-integer searches.
    
    # Let's start with the schoolbook and try to reduce.
    
    # First, build the schoolbook decomposition
    rank_schoolbook = n * m * p
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
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # We can use the fact that some entries can be combined.
    
    # Approach: Try ranks from lower bounds upward.
    # Lower bound: The tensor has a certain structure. For n x m times m x p,
    # the rank is at least max(n*m, m*p, n*p) in many cases, but actually the
    # tensor rank of matrix multiplication is a deep problem.
    
    # For 3x5 * 5x5, let's try to search for ranks below 75.
    
    # A practical approach: use the fact that we can set up a system of equations.
    # For a given rank r, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # and = 0 for invalid ones (but in this case, all (i,j,k) with i<n, j<m, k<p are valid)
    
    # Wait, the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 75 equations in 3*r unknowns (with the constraint
    # that they factorize as products).
    
    # Let's try a different approach: use known algorithms or try specific structures.
    
    # For matrix multiplication, one can use the fact that the tensor is "structured".
    # Let's try to find a decomposition with rank < 75.
    
    # Try rank values from a reasonable lower bound up to 74.
    # Lower bound: at least m = 5 (since we need to capture the m-dimensional sum)
    # Actually, the rank must be at least max(n*m, m*p, p*n) / something... 
    # Let's just try from 25 upward (a reasonable heuristic lower bound).
    
    # Actually, let's think about this differently. The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where e_i are basis vectors. This is the "all-ones" tensor of shape (n,m,p).
    
    # The rank of the all-ones tensor [1]_{n,m,p} is known to be min(n,m,p) in some cases?
    # No, that's not right either.
    
    # Let's just do a computational search. For each candidate rank r, we try to solve
    # the system. We'll use a simple approach: assume integer/half-integer values and
    # use a randomized search.
    
    # Actually, let's try a smarter approach. The all-ones tensor can be written as:
    # T[i,j,k] = 1 = (1/n) * sum_{a=0}^{n-1} exp(2πi*a*i/n) * exp(-2πi*a*j/n) ... no that's for DFT.
    
    # For real/rational decompositions, let's try:
    # If we use rank r, we need r vectors u^(c), v^(c), w^(c) such that
    # sum_c u_i^(c) * v_j^(c) * w_k^(c) = 1 for all i,j,k.
    
    # One simple decomposition: use rank = 1 with u_i = 1, v_j = 1, w_k = 1. 
    # Then sum_c u_i v_j w_k = 1*1*1 = 1. That works!
    
    # Wait, that's rank 1! Let me verify:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    # So the rank is 1? That seems too good. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # The tensor for matrix multiplication C = AB is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 if we're computing the (i,k) entry using (i,j) and (j,k).
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # And indeed, T[i,j,k] = 1 = 1 * 1 * 1, so rank 1 works.
    
    # But wait, is that the correct tensor? In tensor decomposition of matrix multiplication,
    # the tensor is usually defined such that the multiplication is:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = ... hmm.
    
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So the product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all valid (i,j,k) to represent the matrix multiplication tensor.
    
    # So yes, rank 1 with all ones works.
    
    # But that seems trivially easy. Let me double-check by considering if there's a constraint
    # I'm missing. The problem says "integer/half-integer entries". 1 is an integer.
    
    # So the answer is rank 1?
    
    # Let me verify with a simple case: 2x2 * 2x2.
    # T[i,j,k] = 1 for i,j,k in {0,1}.
    # Rank 1: U[i*2+j, 0] = 1, V[j*2+k, 0] = 1, W[k*2+i, 0] = 1.
    # sum_c U[i*2+j,c] V[j*2+k,c] W[k*2+i,c] = 1*1*1 = 1. Correct.
    
    # So the rank is 1 for any matrix multiplication tensor of the form "all ones".
    
    # This makes sense because the matrix multiplication tensor (in this indexing) is just
    # the all-ones tensor, which has rank 1.
    
    U_rank1 = [[1] for _ in range(n * m)]
    V_rank1 = [[1] for _ in range(m * p)]
    W_rank1 = [[1] for _ in range(p * n)]
    
    return {"U": U_rank1, "V": V_rank1, "W": W_rank1}
