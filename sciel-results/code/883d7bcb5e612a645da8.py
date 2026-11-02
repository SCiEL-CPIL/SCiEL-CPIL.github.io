"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 28
    rank_school = n * m * p
    
    # We'll try to find a lower rank decomposition
    # Start with schoolbook and try to reduce rank
    
    # The tensor T[i, j, k] for C[i,k] += A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (14, rank)  
    # W shape: (p*n, rank) = (14, rank)
    # Convention: W[k*n + i, r]
    
    # Let's try rank 24, 23, 22, ... down to some minimum
    # For each rank, try to find a valid decomposition
    
    # First, let's try a structured approach:
    # Group the 14 B-entries into groups and use Strassen-like tricks
    
    # Actually, let me try a direct numerical search for small rank reductions
    # using the fact that we can express some outputs as combinations
    
    # Strategy: Start from schoolbook and try to eliminate terms by finding
    # linear dependencies
    
    # For a rank-r decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,j,k] 
    # where delta is 1 if the term A[i,j]*B[j,k] appears in C[i,k], else 0
    
    # More precisely: the tensor T has T[i, j, k] = 1 if output C[i,k] 
    # includes A[i,j]*B[j,k], which is always true for valid (i,j,k)
    
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is just the all-ones tensor of shape (n, m, p) = (2, 2, 7)
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # M[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # For standard matrix multiplication, M[i,j,k] = 1 for all valid i,j,k
    # So the tensor is indeed all ones of shape (2,2,7)
    
    # The rank of the all-ones tensor of shape (a,b,c) is min(a,b,c) if...
    # No, that's not right either. The rank of the all-ones tensor J_{a,b,c} 
    # where J[i,j,k]=1 is actually more subtle.
    
    # Actually, for the matrix multiplication tensor, it's not all ones.
    # Let me re-read the problem.
    
    # The tensor for multiplying A (n×m) by B (m×p) to get C (n×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i, j, k] = 1 (since each term A[i,j]*B[j,k] 
    # appears exactly once in C[i,k])
    
    # So T is a 3-tensor of shape (n, m, p) with all entries = 1.
    # We want to decompose this as sum_r u_r ⊗ v_r ⊗ w_r where:
    # u_r is in R^n (indexed by i), v_r in R^m (indexed by j), w_r in R^p (indexed by k)
    # Wait, but the interface uses different indexing.
    
    # Let me re-read the interface:
    # U: (n*m, rank) - indexed by (i, j) -> i*m + j
    # V: (m*p, rank) - indexed by (j, k) -> j*p + k
    # W: (p*n, rank) - indexed by (k, i) -> k*n + i
    
    # The decomposition should satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is the all-ones tensor of shape (n*m, m*p, p*n) but with the constraint
    # that the (i*m+j, j*p+k, k*n+i) entry is 1 and others are 0? No...
    
    # Actually, I think the tensor is defined such that:
    # For each r, the contribution to C[i,k] from term r is:
    #   sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we want this to equal sum_j A[i,j]*B[j,k] for all i,k
    # Which means for each i,k: sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
    
    # For this to hold for ALL A and B, we need:
    # For each r, define u_r[i,j] = U[i*m+j,r], v_r[j,k] = V[j*p+k,r], w_r[i,k] = W[k*n+i,r]
    # Then sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = delta_{i,k matches} 
    # Hmm, this is getting confusing.
    
    # Let me think of it as: the bilinear map (A, B) -> C where C = A*B
    # is a tensor in (R^{n*m}*) ⊗ (R^{m*p}*) ⊗ R^{n*p}
    # The rank is the minimum number of rank-1 tensors that sum to it.
    
    # A rank-1 tensor has the form u ⊗ v ⊗ w where u in R^{n*m}, v in R^{m*p}, w in R^{n*p}
    # And the condition is: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = delta_{i,k} * (something)
    
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T satisfies: T[i, j, k] = 1 (for the coefficient of A[i,j]*B[j,k] in C[i,k])
    # But T is a 3-tensor with indices (i,j) for A, (j,k) for B, and (i,k) for C
    # T[(i,j), (j,k), (i,k)] = 1
    
    # So in the flattened indexing:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And T is 0 elsewhere.
    
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    # = 1 if the indices are consistent (i.e., the j in U matches the j in V, 
    #   and the i in W matches the i in U, and the k in W matches the k in V)
    # = 0 otherwise
    
    # This is a complex constraint. Let me just use the schoolbook as a baseline
    # and try to improve it numerically.
    
    # For now, return schoolbook to ensure correctness
    rank = rank_school
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
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
