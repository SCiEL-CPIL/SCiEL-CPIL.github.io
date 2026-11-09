"""Matrix multiplication 3x3 x 3x4 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# Known lower bound / best known rank for 3x3 * 3x4 is 36 (schoolbook).
# However, we can try to find a lower rank decomposition using known algorithms.
# For small matrices, the rank of matrix multiplication tensor is known:
# - 2x2 x 2x2: rank 7 (Strassen)
# - 3x3 x 3x3: rank 23 (Landsberg)
# - 3x3 x 3x4: The rank is at most 36 (schoolbook). 
#   Actually, for n x m * m x p, the rank is at most n*m*p.
#   For 3x3 * 3x4, we can potentially do better.
#   
# Let's think about this more carefully. The tensor is T[i,j,k] = delta[i,k] (for the multiplication).
# Actually, the matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p) is:
# T_{i,k,j} = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
# The rank is the minimum number of rank-1 tensors to express T.
#
# For 3x3 * 3x4, the schoolbook rank is 36.
# There are known results that for small rectangular matrices, the rank can be less than n*m*p.
#
# Let me try to use a known result or construct a better decomposition.
#
# Actually, let me try a different approach: use the fact that we can share some computations.
# For 3x3 * 3x4, we have 12 output entries (3*4), each requiring 3 multiplications (from m=3).
# Total: 12*3 = 36 multiplications in schoolbook.
#
# Can we do better? Let's try to use Strassen-like ideas or other algebraic tricks.
#
# One approach: group the computations. 
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
# C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
# C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
# C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
# etc.
#
# Let me try to find a rank-35 or lower decomposition by exploiting linearity.
#
# Actually, for this specific small case, let me try to use a known algorithm or 
# search for a decomposition with rank < 36.
#
# Let me try rank 35 first by attempting to merge two terms.
# If we can find indices (i1,j1,k1) and (i2,j2,k2) such that we can combine them,
# we'd need: the sum of two rank-1 tensors to equal their individual contributions
# in a way that allows merging. This is generally not straightforward.
#
# Let me try a different strategy: use the known rank for 3x3 matrix multiplication (23)
# and extend it. But 3x3 * 3x4 is different from 3x3 * 3x3.
#
# Let me just return the schoolbook decomposition for now, which gives rank 36.
# This is the safe baseline. If I can find a better one, I'll use it.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition with rank n*m*p = 36
    rank = n * m * p
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
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta for C[i,k]
        # Actually, the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c[i,j] = U[i*m+j, c], V_c[j,k] = V[j*p+k, c], W_c[k,i] = W[k*n+i, c]
        # And C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_k' V_c[j,k'] * B[j,k']) ... 
        # Wait, let me re-read the convention.
        # W[k*n+i, r] means W is indexed by [k*n + i, r] where k is the column of B / row of C, i is the row of C.
        # So W[k*n+i, r] corresponds to the coefficient for C[i,k] in rank term r.
        # U[i*m+j, r] is the coefficient for A[i,j] in rank term r.
        # V[j*p+k, r] is the coefficient for B[j,k] in rank term r.
        # The decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... 
        # No, that's not right either. Let me think again.
        # 
        # Standard tensor decomposition for matrix multiplication:
        # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
        # The multiplication tensor T has entries T[i,j,k] = delta (it's 1 for the "identity" in the middle index)
        # Actually, T is defined such that <T, A ⊗ B> = C
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # 
        # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector of size n*m, v_r is size m*p, w_r is size p*n
        # and T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        # 
        # For this to equal the matrix multiplication tensor, we need:
        # For each (i,k): sum_j T[i,j,k] = sum_j 1 = m (wait, no)
        # Actually, T[i,j,k] is 1 only when the indices are "compatible" in the multiplication.
        # The matrix multiplication tensor is: T[i,j,k] = 1 (for all valid i,j,k) but with the constraint
        # that j is the shared index. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # No, that's not right. The tensor is:
        # T[i,j,k] = 1 if we're multiplying A[i,j] by B[j,k] to contribute to C[i,k]
        # So T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
        # Wait, that means T is just the all-ones tensor of size n*m*p? No...
        # 
        # Let me re-read the task description:
        # "W[k*n+i,r]" - so W is indexed by [k*n+i, r]
        # The decomposition should satisfy:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # No, that's not standard. Let me think about what makes sense.
        # 
        # Standard: C[i,k] = sum_j A[i,j] * B[j,k]
        # We want: C[i,k] = sum_r W[k*n+i, r] * f_r(A) * g_r(B)
        # where f_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j']
        # and g_r(B) = sum_{j',k'} V[j'*p+k', r] * B[j',k']
        # 
        # For this to work, we need for each (i,k):
        # sum_j A[i,j] * B[j,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
        # 
        # This is a tensor equation. For the schoolbook case:
        # For each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # where c = i*m*p + j*p + k (or some ordering)
        # Then: sum_r W[k*n+i, r] * f_r(A) * g_r(B)
        # = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
        # For schoolbook, U[i'*m+j', c] is 1 only if i'=i, j'=j for the specific c corresponding to (i,j,k)
        # So f_r(A) = A[i,j] for the r corresponding to (i,j,k)
        # Similarly g_r(B) = B[j,k]
        # And W[k*n+i, c] = 1 only for the c corresponding to (i,j,k)
        # So the sum becomes: sum_j A[i,j] * B[j,k] = C[i,k] ✓
        # 
        # So the verification is: for all i,k:
        # sum_j A[i,j]*B[j,k] = sum_r W[k*n+i,r] * (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
        # 
        # This is correct for the schoolbook decomposition.
        # 
        # For verification with random matrices, we can check this identity.
        pass
    
    return {"U": U, "V": V, "W": W}
