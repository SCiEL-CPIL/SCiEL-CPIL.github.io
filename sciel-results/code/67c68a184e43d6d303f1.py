"""Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# The matrix multiplication tensor T[i,j,k] = e_{i,k} (identity in the output index)
# For (2,5,5): n=2, m=5, p=5
# U has shape (n*m, rank) = (10, rank)
# V has shape (m*p, rank) = (25, rank)
# W has shape (p*n, rank) = (10, rank)
#
# The tensor equation is:
#   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
#
# This is a 3-way tensor. The rank of the matrix multiplication tensor M_{n,m,p}
# is known for small cases. For (2,5,5), the schoolbook rank is 2*5*5 = 50.
#
# Known results: The rank of matrix multiplication for (n,m,p) where at least one
# dimension is small has been studied. For (2,5,5), we need to find if there's a
# decomposition with rank < 50.
#
# The tensor T_{i,j,k} = delta_{i,k} can be viewed as: for each (i,k) pair where i==k,
# we have a "1" in the j-th position summed over r.
#
# Let's think about this differently. The tensor is:
#   T = sum_{i=0}^{n-1} e_i ⊗ I_m ⊗ e_i   (in the appropriate index ordering)
#
# Actually, T[i,j,k] = 1 if i==k, else 0.
#
# One approach: use the fact that T can be written as a sum of rank-1 tensors.
# A rank-1 tensor has the form u_i * v_j * w_k.
#
# For the (2,5,5) case, let's try to find a low-rank decomposition.
#
# Key insight: T[i,j,k] = delta_{i,k}. We can write:
#   T = sum_{r} u_r ⊗ v_r ⊗ w_r
#
# where u_r ∈ R^n, v_r ∈ R^m, w_r ∈ R^p.
#
# This means: sum_r u_r[i] * v_r[j] * w_r[k] = delta_{i,k}
#
# For n=2, p=5, m=5:
#   For (i,k) with i≠k: sum_r u_r[i] * v_r[j] * w_r[k] = 0 for all j
#   For (i,k) with i=k: sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all j
#
# The "for all j" part means that for each (i,k), the vector (over j) of 
# sum_r u_r[i] * v_r[j] * w_r[k] must be either all zeros (if i≠k) or all ones (if i=k).
#
# This is equivalent to saying that the matrix A_r = u_r[i] * w_r[k] * v_r (as a row vector)
# summed over r gives specific patterns.
#
# Let's define for each r: a_r[i,k] = u_r[i] * w_r[k] (an n×p matrix, rank 1)
# and b_r[j] = v_r[j] (an m-vector).
#
# Then T[i,j,k] = sum_r a_r[i,k] * b_r[j].
#
# We need: sum_r a_r[i,k] * b_r[j] = delta_{i,k} for all i,j,k.
#
# This means that for each fixed j, the matrix (sum_r a_r * b_r[j]) = I_n (n×n identity).
# Wait, that's not quite right because b_r[j] is a scalar.
#
# Actually: sum_r (u_r[i] * w_r[k]) * v_r[j] = delta_{i,k}
#
# Let M_r = u_r * w_r^T (outer product, n×p matrix).
# Then: sum_r M_r[i,k] * v_r[j] = delta_{i,k}
#
# For this to hold for all j, we need:
#   For each j: sum_r v_r[j] * M_r = I_n  (as an n×p matrix? No, I_n is n×n)
#
# Wait, the RHS is delta_{i,k} which is 1 if i=k. But k ranges over p=5 values and i over n=2.
# So delta_{i,k} is only defined/meaningful when we interpret it as: the (i,k) entry is 1 if i=k, 0 otherwise.
# Since i ∈ {0,1} and k ∈ {0,1,2,3,4}, delta_{i,k} = 1 if i=k (i.e., (0,0) and (1,1)), 0 otherwise.
#
# So the condition is:
#   sum_r v_r[j] * M_r[i,k] = delta_{i,k}  for all i∈{0,1}, j∈{0,...,4}, k∈{0,...,4}
#
# This must hold for all j. So for each j, the matrix sum_r v_r[j] * M_r must equal the fixed matrix D where D[i,k] = delta_{i,k}.
#
# D = [[1,0,0,0,0],
#      [0,1,0,0,0]]
#
# So we need: for all j, sum_r v_r[j] * M_r = D.
#
# This means the vectors (M_0, M_1, ..., M_{R-1}) in R^{n*p} = R^{10} with coefficients v_r[j] must always sum to D.
#
# This is possible if all v_r are the same vector (say all-ones), but then we'd need sum_r M_r = D, and each M_r is rank 1.
# The rank of D is 2, so we need at least 2 rank-1 matrices to sum to D. In fact, D = e_0 ⊗ e_0^T + e_1 ⊗ e_1^T, so we can use 2 terms with v_r = (1,1,1,1,1).
#
# But wait - if v_r is the same for all r, then the decomposition is:
#   U[i,j] = u_r[i] (depends on r, not j)
#   V[j,k] = v_r[j] (depends on r and j)
#   W[k,i] = w_r[k] (depends on r, not i... wait)
#
# Let me re-index properly.
#
# U has shape (n*m, rank) = (10, R). U[i*m+j, r] = u_r[i] (independent of j)?
# V has shape (m*p, rank) = (25, R). V[j*p+k, r] = v_r[j] * something? 
#
# Hmm, let me reconsider. The constraint is:
#   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
#
# If I set U[i*m+j, r] = a_r[i] (independent of j), V[j*p+k, r] = b_r[j,k], W[k*n+i, r] = c_r[k,i],
# then sum_r a_r[i] * b_r[j,k] * c_r[k,i] = delta_{i,k}.
#
# This is getting complex. Let me just try a direct numerical search for small ranks.
#
# Actually, let me think about what rank is achievable. The trivial bound is n*m*p = 50.
# But we can do better. The tensor T has "slice rank" or "subrank" considerations.
#
# For the matrix multiplication tensor, the rank is known to be strictly less than n*m*p for many cases.
# For (2,5,5), let's try to find a good decomposition.
#
# Approach: Set up a system of equations. We have R rank-1 terms. Each term has:
#   U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for r=0..R-1
#
# Total unknowns: R*(n*m + m*p + p*n) = R*(10+25+10) = 45R
#
# Constraints: n*m*p = 50 equations (one for each (i,j,k) triple).
#
# But we also have the scaling freedom: we can scale U_r by α, V_r by β, W_r by γ with α*β*γ=1.
# So effectively 2 degrees of freedom per term in scaling.
#
# For a linear approach, we can fix one entry per term to break scaling symmetry.
#
# Let me try a different approach: use the structure of the problem.
#
# T[i,j,k] = delta_{i,k}. Note that for fixed i,k, the value is independent of j.
# This means the tensor is "separable" in the j-index in a specific way.
#
# We can write: T = sum_{i=0}^{1} E_i where E_i[i',j,k] = 1 if i'=i and k=i, else 0.
# E_i = e_i ⊗ (1,1,1,1,1) ⊗ e_i  (as a rank-1 tensor? No, because the middle component is a constant vector)
#
# Wait: E_i[i',j,k] = delta_{i',i} * 1 * delta_{k,i} = (e_i[i'] * 1 * e_i[k])
# So E_i = e_i ⊗ 1_m ⊗ e_i is indeed a rank-1 tensor!
#
# So T = E_0 + E_1 = (e_0 ⊗ 1_5 ⊗ e_0) + (e_1 ⊗ 1_5 ⊗ e_1)
#
# This gives a rank-2 decomposition!
#
# Let's verify:
#   Term 0: U[i*m+j, 0] = e_0[i] = 1 if i=0 else 0
#            V[j*p+k, 0] = 1 (for all j,k)
#            W[k*n+i, 0] = e_0[k] = 1 if k=0 else 0
#
#   Term 1: U[i*m+j, 1] = e_1[i] = 1 if i=1 else 0
#            V[j*p+k, 1] = 1 (for all j,k)
#            W[k*n+i, 1] = e_1[k] = 1 if k=1 else 0
#
# Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
#   = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] + U[i*m+j,1]*V[j*p+k,1]*W[k*n+i,1]
#   = (1 if i=0 else 0) * 1 * (1 if k=0 else 0) + (1 if i=1 else 0) * 1 * (1 if k=1 else 0)
#   = (1 if i=0 and k=0 else 0) + (1 if i=1 and k=1 else 0)
#   = delta_{i,k}  (since i,k ∈ {0,1} and the delta is 1 only when i=k)
#
# Yes! Rank 2 works!
#
# Wait, but this seems too good to be true. Let me double-check the dimensions.
# n=2, m=5, p=5.
# U: (n*m, rank) = (10, 2). U[i*m+j, r] where i∈{0,1}, j∈{0,...,4}.
# V: (m*p, rank) = (25, 2). V[j*p+k, r] where j∈{0,...,4}, k∈{0,...,4}.
# W: (p*n, rank) = (10, 2). W[k*n+i, r] where k∈{0,...,4}, i∈{0,1}.
#
# The trace convention is W[k*n+i, r]. So W is indexed by (k, i) with k∈{0,...,p-1}, i∈{0,...,n-1}.
#
# For term 0:
#   U[i*5+j, 0] = 1 if i=0, else 0. So U[0:5, 0] = 1, U[5:10, 0] = 0.
#   V[j*5+k, 0] = 1 for all j,k. So V[:, 0] = 1.
#   W[k*2+i, 0] = 1 if k=0, else 0. So W[0:2, 0] = 1, W[2:10, 0] = 0.
#
# For term 1:
#   U[i*5+j, 1] = 1 if i=1, else 0. So U[0:5, 1] = 0, U[5:10, 1] = 1.
#   V[j*5+k, 1] = 1 for all j,k. So V[:, 1] = 1.
#   W[k*2+i, 1] = 1 if k=1, else 0. So W[0:2, 1] = 0, W[2:4, 1] = 1, W[4:10, 1] = 0.
#
# Let's verify a few entries:
# (i=0, j=2, k=0): 
#   U[0*5+2, 0]*V[2*5+0, 0]*W[0*2+0, 0] + U[0*5+2, 1]*V[2*5+0, 1]*W[0*2+0, 1]
#   = U[2,0]*V[10,0]*W[0,0] + U[2,1]*V[10,1]*W[0,1]
#   = 1*1*1 + 0*1*0 = 1. And delta_{0,0} = 1. ✓
#
# (i=0, j=2, k=1):
#   U[2,0]*V[11,0]*W[2,0] + U[2,1]*V[11,1]*W[2,1]
#   = 1*1*0 + 0*1*0 = 0. And delta_{0,1} = 0. ✓
#
# (i=1, j=3, k=1):
#   U[8,0]*V[18,0]*W[3,0] + U[8,1]*V[18,1]*W[3,1]
#   = 0*1*0 + 1*1*1 = 1. And delta_{1,1} = 1. ✓
#
# (i=1, j=3, k=0):
#   U[8,0]*V[15,0]*W[1,0] + U[8,1]*V[15,1]*W[1,1]
#   = 0*1*1 + 1*1*0 = 0. And delta_{1,0} = 0. ✓
#
# (i=1, j=0, k=2):
#   U[5,0]*V[2,0]*W[5,0] + U[5,1]*V[2,1]*W[5,1]
#   = 0*1*0 + 1*1*0 = 0. And delta_{1,2} = 0. ✓
#
# Great, rank 2 is valid!

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    rank = 2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: corresponds to i=0, k=0
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1 if i == 0 else 0
            U[i * m + j][1] = 1 if i == 1 else 0
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
            V[j * p + k][1] = 1
    
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1 if k == 0 else 0
            W[k * n + i][1] = 1 if k == 1 else 0
    
    return {"U": U, "V": V, "W": W}
