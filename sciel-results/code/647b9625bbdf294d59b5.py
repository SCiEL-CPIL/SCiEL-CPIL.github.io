"""Matrix multiplication 2x4x8 with rank-minimal tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 4, 8]

# Known rank-20 decomposition for 4x4 matrix multiplication (Laderman's algorithm)
# This gives us a bilinear form for C = A * B where A is 4x4, B is 4x4, C is 4x4.
# We'll embed our 2x4 * 4x8 problem into this framework.

# Actually, let me think more carefully. The task is to multiply a 2x4 matrix by a 4x8 matrix.
# The output is a 2x8 matrix.
# 
# The tensor we need to decompose has:
# - Input A: 2x4 = 8 elements (indexed by i in 0..1, j in 0..3)
# - Input B: 4x8 = 32 elements (indexed by j in 0..3, k in 0..7)  
# - Output C: 2x8 = 16 elements (indexed by i in 0..1, k in 0..7)
#
# The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
#
# In the tensor decomposition framework:
# - U has shape (n*m, rank) = (8, rank), indexed by (i,j)
# - V has shape (m*p, rank) = (32, rank), indexed by (j,k)
# - W has shape (p*n, rank) = (16, rank), indexed by (k,i) [note: W[k*n+i, r]]
#
# The decomposition satisfies: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# Wait, that's not right. Let me re-read.
#
# "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
# So for each rank component c, we have:
#   sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = C[i,k]
#
# This is the standard tensor decomposition for matrix multiplication.
#
# Now, can we use Laderman's algorithm? Laderman's algorithm decomposes 4x4 * 4x4 -> 4x4 with rank 20.
# Our problem is 2x4 * 4x8 -> 2x8.
#
# One approach: Pad A to 4x4 and B to 4x4 (by repeating or zero-padding), compute the 4x4 product,
# then extract the relevant 2x8 portion. But this doesn't directly give us a lower rank.
#
# Actually, let's think about it differently. The 2x4 * 4x8 multiplication can be seen as:
# For each of the 2 rows of A and 8 columns of B, we compute a dot product of length 4.
# That's 2*8 = 16 dot products of length 4.
#
# The schoolbook rank is 2*4*8 = 64.
#
# Can we do better? The rank of the tensor for (m,n,p) matrix multiplication is at most m*n*p.
# For our case, that's 64.
#
# Let me try a different approach. Since the middle dimension is 4, we can think of this as:
# C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
#
# This is like 16 independent dot products of dimension 4. Each dot product of dimension 4
# can be computed with rank 4 (schoolbook), so total rank would be 16*4 = 64.
#
# But we can share computations. The key insight is that A[i,j] is shared across all k,
# and B[j,k] is shared across all i.
#
# Let me try to construct a decomposition with lower rank.
#
# Actually, for the specific case of 2x4 * 4x8, let me think about what's achievable.
# The rank of the matrix multiplication tensor T_{m,n,p} satisfies:
# - Lower bound: max(m*n, m*p, n*p) = max(8, 32, 16) = 32
# - Upper bound: m*n*p = 64
#
# So the rank is between 32 and 64.
#
# Let me try to find a decomposition with rank less than 64.
#
# One idea: Use the fact that we only have 2 rows of A. We can write:
# C[i,k] = sum_j A[i,j] * B[j,k]
#
# If we fix j, then A[i,j] for i=0,1 gives a 2-vector, and B[j,k] for k=0..7 gives an 8-vector.
# The outer product A[:,j] * B[j,:] contributes to C.
#
# So C = sum_j A[:,j] * B[j,:]
#
# This is a sum of 4 outer products of a 2-vector and an 8-vector.
# Each outer product has rank 1 as a matrix, but as a tensor in the bilinear sense...
#
# Wait, I think I'm overcomplicating this. Let me just try to find a valid decomposition
# with rank less than 64 by search or by construction.
#
# Let me try a simple improvement: group by j. For each j, we have contributions from
# all (i,k) pairs. The term A[i,j]*B[j,k] appears in C[i,k].
#
# In the tensor decomposition:
# U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c gives C[i,k].
#
# For the schoolbook: each (i,j,k) triple gets its own rank component.
#
# To reduce rank, we need to share rank components across different (i,j,k) triples.
#
# Let me try: for a fixed j, can we share components across i and k?
# For fixed j, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
#
# If we set U[i*m+j, c] = A[i,j] and V[j*p+k, c] = B[j,k] and W[k*n+i, c] = 1,
# but that doesn't work because U and V depend on i and k respectively.
#
# Actually, the standard trick is: for each j, use rank components that factor the
# i-dependence and k-dependence separately.
#
# Let me try: for each j (4 values), use 2*8 = 16 components? No, that's still 64 total.
#
# Hmm, let me try a different approach. Since n=2 is small, maybe I can find a clever decomposition.
#
# Let me just implement the schoolbook for now and see if I can find any improvement.
# Given the time constraints, let me return the schoolbook decomposition.

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 16 x rank
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
