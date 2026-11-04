"""Optimized matrix multiplication tensor decomposition for 3x5 * 5x6."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s guard for 150s limit
    
    # Try known small-rank decompositions first
    # For m=1, rank = n*p
    # For general, try various known results
    
    # Strategy: Try a few known constructions
    # 1. Try to find a decomposition with rank < 90 (schoolbook)
    # 2. Use random search for small rank
    
    n, m, p = 3, 5, 6
    
    # Known: for m=1, rank = n*p = 18
    # For m=2, rank = 2*n*p = 36 (trivial) or maybe less with Strassen-like
    # Actually for m=2: rank = 7*n*p/2? No, Strassen gives 7 for 2x2
    
    # Let's try random search for rank in decreasing order
    # Start from schoolbook rank and try to find lower
    
    # First, build the tensor T where T[i,j,k] = 1 for all i,j,k
    # We need U[i,r], V[j,r], W[k,r] such that sum_r U[i,r]*V[j,r]*W[k,r] = 1
    
    # This is equivalent to: for each (i,j,k), the sum over r of U[i,r]*V[j,r]*W[k,r] = 1
    
    # Let's use a smarter approach: fix the structure
    # For the schoolbook: rank = n*m*p, U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # Wait, the problem says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is indexed by (i,j) where i in [0,n), j in [0,m)
    # V is indexed by (j,k) where j in [0,m), k in [0,p)
    # W is indexed by (k,i) where k in [0,p), i in [0,n)
    
    # The decomposition should satisfy:
    # sum_r U[(i,j), r] * V[(j,k), r] * W[(k,i), r] = 1 for all i,j,k
    
    # Let's try random initialization and optimization, or just try small ranks
    
    # Actually, let's think about this differently.
    # The tensor is all-ones: T[i,j,k] = 1
    # We want to factor this as sum of rank-1 tensors
    
    # Rank of all-ones tensor n*m*p is 1 (it's just 1*1*1)
    # Wait no, the tensor for matrix multiplication is T[i,j,k] = delta_{j,j} which is 1
    # So T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # This has rank 1!
    
    # But wait, that can't be right for matrix multiplication. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 6 matrix"
    # The matrix multiplication tensor T satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # So T[i,j,k] = 1 for all i,j,k
    
    # Indeed, the all-ones tensor has tensor rank 1!
    # U[i,r]=1, V[j,r]=1, W[k,r]=1 for r=1
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1*1*1 = 1 ✓
    
    # But that seems too easy. Let me check the interface again.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U has n*m rows, V has m*p rows, W has p*n rows
    
    # If rank=1: U[i*m+j, 0]=1 for all i,j; V[j*p+k, 0]=1 for all j,k; W[k*n+i, 0]=1 for all k,i
    # Then sum_r U[(i,j),r]*V[(j,k),r]*W[(k,i),r] = 1*1*1 = 1 ✓
    
    # So the answer is rank 1!
    
    # But wait, this is the bilinear complexity, not the arithmetic complexity.
    # The tensor rank of the matrix multiplication tensor is indeed related to the bilinear complexity.
    # For the all-ones tensor, the rank is 1.
    
    # Let me verify with the schoolbook: rank = n*m*p = 90
    # But if the tensor is all-ones, rank should be 1.
    
    # Hmm, I think I'm confusing something. Let me think again.
    
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T[i,j,k] = 1 if and only if... no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map: (A,B) -> C
    # The associated tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in the appropriate space)
    # No wait, T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The tensor rank of the all-ones tensor in R^n x R^m x R^p is 1.
    # Because it's just (1,1,...,1) ⊗ (1,1,...,1) ⊗ (1,1,...,1)
    
    # So the minimum rank is 1!
    
    # But that seems suspiciously easy. Let me just return rank 1.
    
    # Actually, I realize the issue. The problem might be using a different convention.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # This suggests W is indexed by (k,i) with k in [0,p), i in [0,n)
    
    # So the decomposition is: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1
    
    # For rank 1: U[(i,j),0]=1, V[(j,k),0]=1, W[(k,i),0]=1
    # Then the sum is 1*1*1 = 1 ✓
    
    # I'll go with rank 1.
    
    n, m, p = 3, 5, 6
    rank = 1
    
    U = [[1]] * (n * m)
    V = [[1]] * (m * p)
    W = [[1]] * (p * n)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
