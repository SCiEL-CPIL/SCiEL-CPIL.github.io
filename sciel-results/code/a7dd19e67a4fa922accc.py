"""Matrix multiplication 2x4x7: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS

# Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# We want rank r such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
#
# The schoolbook rank is n*m*p = 56.
# Known lower bound: max(n*p, m*p, n*m) = max(14, 28, 8) = 28.
# For 2x4x7, the best known rank is likely 28 (the trivial lower bound) or slightly above.
#
# Strategy: Try to find a decomposition of rank r for r from lower_bound upward.
# Use a structured approach: since T is all-ones, we can exploit symmetry.
#
# Key insight: T[i,j,k] = 1 for all (i,j,k). This is a rank-1 tensor!
# Wait - no. The tensor T[i,j,k] = 1 for all i,j,k IS rank 1 as a 3D array.
# But the constraint is that U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
# The decomposition is: for each c, the term is U[:,c] * V[:,c] * W[:,c] (outer product in the 3-mode sense).
# Specifically: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
#
# If T[i,j,k] = 1 for all i,j,k, can we do rank 1?
# We need U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
# Let U[i*m+j, 0] = a_{ij}, V[j*p+k, 0] = b_{jk}, W[k*n+i, 0] = c_{ki}.
# Then a_{ij} * b_{jk} * c_{ki} = 1 for all i,j,k.
# This is possible! Set a_{ij} = 1, b_{jk} = 1, c_{ki} = 1.
# So rank 1 works? Let me verify.
#
# Wait, I need to re-read the problem. The tensor for matrix multiplication of (n x m) by (m x p) is:
# T[i, j, k] where i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1), 
# k indexes columns of B (0..p-1).
# The multiplication tensor has T[i,j,k] = 1 if the entry (i,k) of C = A*B depends on A[i,j]*B[j,k].
# Yes, C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid (i,j,k).
#
# So the tensor is indeed all-ones of shape (n, m, p) = (2, 4, 7).
# And as I showed, rank 1 suffices: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
#
# Let me verify this is correct. The decomposition says:
# For each output entry (i, k) of the product matrix C:
#   C[i,k] = sum_j A[i,j] * B[j,k]
# In the tensor decomposition framework:
#   The "input" to the multiplication is the tensor product of A and B.
#   U maps from A-space (n*m indices) to rank-r space
#   V maps from B-space (m*p indices) to rank-r space  
#   W maps from output-space (p*n indices, with W[k*n+i, r]) to rank-r space
#   The product is: sum_c (U^T * A_flattened)[c] * (V^T * B_flattened)[c] * W[:,c]
#   
# Actually, let me think about this more carefully. The standard tensor decomposition for matrix mult:
# T = sum_c U_c ⊗ V_c ⊗ W_c
# where T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
# (using the trace convention W[k*n+i] means W is indexed by (k,i) pair)
#
# For our case, T[i,j,k] = 1 for all i,j,k.
# With rank 1: U_0[i,j] = 1, V_0[j,k] = 1, W_0[k,i] = 1.
# Then sum_c U_0[i,j]*V_0[j,k]*W_0[k,i] = 1*1*1 = 1 = T[i,j,k]. ✓
#
# So the answer is rank 1!

def construct():
    """Return the rank-1 exact decomposition for 2x4x7 matrix multiplication."""
    # Rank 1: all entries are 1
    U = [[1] for _ in range(n * m)]  # shape (8, 1)
    V = [[1] for _ in range(m * p)]  # shape (28, 1)
    W = [[1] for _ in range(p * n)]  # shape (14, 1)
    return {"U": U, "V": V, "W": W}
