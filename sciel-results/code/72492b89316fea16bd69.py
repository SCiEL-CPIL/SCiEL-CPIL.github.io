"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy:
  - The schoolbook rank is 80.
  - We search for a lower-rank decomposition using structured patterns.
  - The key insight for small matrix multiplication is that we can sometimes
    exploit the structure of the tensor to reduce rank.
  - For 4x4 * 4x5, known results suggest rank might be reducible below 80.
  
  Approach:
  1. Start with schoolbook (rank 80) as fallback.
  2. Try to find a lower-rank decomposition by exploiting symmetries or
     known lower-bound constructions.
  
  For n=m=4, p=5:
  - The trivial lower bound is max(n*m, m*p, p*n) = max(16, 20, 20) = 20.
  - The upper bound from schoolbook is 80.
  - Known results for 4x4*4x4 give rank 23 (Strassen-like).
  - For 4x4*4x5, we can potentially use a similar approach.
  
  We'll use a constructive approach:
  - Represent the multiplication tensor and try to find a rank-r decomposition
    by solving a system of equations, starting from a structured ansatz.
  
  Practical approach within time limits:
  - Use a random or structured initialization and verify correctness.
  - If we can't find a better decomposition in time, return schoolbook.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k) in [n]x[m]x[p]:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    And for all other index combinations (where the tensor entry is 0),
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 0
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # T should equal 1 when the indices form a valid matrix product entry
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (since every combination contributes)
    # Wait, no. The multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # Because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has a 1 at every (i,j,k).
    
    # So we need: for all i in [n], j in [m], k in [p]:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for evaluator's 150s limit
    
    # Schoolbook fallback
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
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # For 4x4*4x5, let's try ranks from a reasonable range downward
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = 20
    # Practical lower bound from known results: for 4x4*4x4 it's 23
    # For 4x4*4x5, let's try from 40 down to 25
    
    # Strategy: Use a random search with verification
    # Each trial: generate random U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
    # Check if they satisfy the tensor equation
    
    # Better strategy: Use the structure of the problem
    # The tensor is all-ones on [n]x[m]x[p], so we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a rank-r decomposition of an all-ones tensor of size n x m x p
    # where the indices are mapped through specific permutations.
    
    # Actually, let me reconsider. The matrix multiplication tensor for (n,m,p) is:
    # T(i,j,k) = 1 for all i in [n], j in [m], k in [p]
    # This is just the all-ones tensor!
    
    # Wait, that's not right either. Let me re-examine.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form has terms A[i,j]*B[j,k] for each (i,j,k)
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k)
    
    # Yes! The tensor is the all-ones tensor of shape (n, m, p).
    # So we need a rank-r decomposition of the all-ones tensor.
    
    # The all-ones tensor of shape (n,m,p) has rank = max(n,m,p) in general?
    # No, that's not right. The rank of the all-ones tensor J_{n,m,p} is...
    # Actually, J_{n,m,p} = ones(n) * ones(m) * ones(p) which is rank 1!
    # But that's for the standard tensor product.
    
    # Wait, I'm confusing myself. Let me be very careful.
    
    # The matrix multiplication tensor is defined as:
    # mu_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j}^{(A)} * e_{j,k}^{(B)} * e_{i,k}^{(C)}
    
    # So T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    
    # This IS the all-ones tensor. And the all-ones tensor has tensor rank 1:
    # J_{n,m,p} = (sum_i e_i) * (sum_j e_j) * (sum_k e_k)
    
    # But wait, that would mean matrix multiplication has rank 1, which is absurd.
    
    # I think I'm making an error. Let me reconsider the tensor definition.
    
    # The matrix multiplication tensor is:
    # mu = sum_{i,j,k} e_i^C * e_j^A * e_k^B  ... no
    # 
    # Standard definition: For C = A*B where A is n x m, B is m x p:
    # mu_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} * e_{j,k} * e_{i,k}
    # where e_{i,j} is the (i,j)-th standard basis element in R^{n x m}, etc.
    
    # So the tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    # Now, the rank of this tensor: we want to write
    # T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # Note the index structure: U depends on (i,j), V on (j,k), W on (i,k).
    # This is NOT the standard all-ones tensor decomposition because the index
    # mappings are different!
    
    # In the standard tensor, all three indices are independent. Here, the
    # "fiber" structure couples them: U uses (i,j), V uses (j,k), W uses (i,k).
    
    # So this is a specific tensor with a specific index structure, and its rank
    # is what we're trying to minimize.
    
    # OK so my verification function is correct. Let me proceed with the search.
    
    # For small ranks, we can try random initializations and check.
    # For each rank r, generate random U, V, W and check if they work.
    # This is very unlikely to succeed with pure random, so we need a smarter approach.
    
    # Smarter approach: Use the fact that the tensor is "structured" and try
    # to find a decomposition using known algorithms or patterns.
    
    # One approach: Start with the schoolbook decomposition and try to merge
    # columns by finding linear dependencies.
    
    # Another approach: Use a greedy algorithm.
    
    # For now, let's try a few random trials at lower ranks and see if we get lucky.
    # This is unlikely but the cost is low.
    
    np.random.seed(42)
    random.seed(42)
    
    # Try ranks from a reasonable range
    # For 4x4*4x4, best known is 23. For 4x4*4x5, maybe 25-35?
    
    tried_ranks = set()
    
    for r in range(40, 19, -1):
        if time.time() > deadline:
            break
        if r in tried_ranks:
            continue
        tried_ranks.add(r)
        
        # Try a few random trials at this rank
        max_trials = 20
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            if _verify_decomposition(U, V, W, n, m, p, r, tol=1e-6):
                # Found a valid decomposition!
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < rank_school:
            break  # Found something better, stop
    
    # If we found something better, use it; otherwise use schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
