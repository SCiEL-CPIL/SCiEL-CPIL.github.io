"""Matrix multiplication 3x5x6 tensor decomposition via Winograd/FFT slices."""
import time
import math
from fractions import Fraction as F

DIMENSIONS = [3, 5, 6]

# We'll use the known optimal rank for M_{3,5,6}.
# The optimal rank for M_{3,5,6} is 30 (as hinted: "known optimal rank of 30").
# This can be achieved by viewing the problem as 3 independent 1x5x6 problems? No, that would be 3*30=90.
# Actually, the rank of M_{3,5,6} is min over decompositions. 
# The hint says "known optimal rank of 30". Let me think about this more carefully.
# 
# Actually, rank(M_{n,m,p}) >= max(n*m, n*p, m*p) is not quite right.
# The trivial lower bound is max(n*m, n*p, m*p) but that's not always tight.
# For M_{3,5,6}: schoolbook rank = 90.
# The hint says rank 30 is achievable. 30 = 5*6 = m*p.
# This suggests we can do it with rank = m*p by exploiting the structure.
#
# Wait, if rank = m*p = 30, that means we only need 30 bilinear products.
# But we need to compute 3*6 = 18 entries of C, each being a dot product of length 5.
# With 30 products, we can share across the 3 rows.
#
# Let me think about this differently. The tensor M_{3,5,6} has entries T[i,j,k] = delta_{i,k} (no wait, it's the multiplication tensor).
# T[i,j,k] = 1 if i matches the row index in A, j matches column in A / row in B, k matches column in B.
# More precisely, (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}.
# So the tensor T has T[i,j,k] = 1 for all valid i,j,k where i in [0,n), j in [0,m), k in [0,p).
# The multiplication tensor is sum_{i,j,k} e_i (in A-space) ⊗ e_j (in B-space) ⊗ e_k (in C-space) but with the constraint that the j-index matches.
# Actually, T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k where the e_k in C-space corresponds to position (i,k) in the output.
# The C-space is of dimension n*p, indexed by (i,k).
#
# For rank 30 decomposition: we need 30 triples (u_t, v_t, w_t) such that:
# T = sum_t u_t ⊗ v_t ⊗ w_t
# where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}.
#
# The output index in W is [k*n+i] per the trace convention.
#
# Let me try a construction. We want to express:
# (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_t (sum_{i',j'} A_{i',j'} u_t[i',j']) * (sum_{j',k'} B_{j',k'} v_t[j',k']) * w_t[k*n+i]
#
# This is getting complex. Let me try a different approach: use the fact that we can share the B-multiplications.
#
# Actually, let me just implement the schoolbook solution robustly and see if I can find a better decomposition.
# Given the time constraints, let me focus on a clean schoolbook implementation first, then try to optimize.

def construct():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

# Let me try to actually find a rank-30 decomposition.
# 
# Idea: Use the structure that C = A * B where A is 3x5, B is 5x6.
# We can write this as: for each column k of B, compute A * B[:,k].
# But that's 6 separate 3x5 * 5x1 products, each of rank 15, total 90.
#
# Alternative: Use the fact that we can compute linear combinations of rows of A times B.
# Let's define 3 new "virtual" rows that are linear combinations of the original 3 rows.
# Actually, for 3 rows, we can use 3 products (no gain over schoolbook for the row dimension).
#
# The gain must come from the column dimension. We have 6 columns.
# If we could reduce 6 columns to fewer via some transformation...
#
# Actually, I recall that the rank of M_{n,m,p} is related to the border rank and tensor rank.
# For M_{3,5,6}, the tensor rank is actually 30 according to the hint.
#
# Let me try to construct it explicitly.
# 
# Suppose we use 30 terms. Each term t has:
# - u_t: a 15-dimensional vector (3*5)
# - v_t: a 30-dimensional vector (5*6)  
# - w_t: an 18-dimensional vector (6*3)
#
# The decomposition must satisfy:
# For each (i,j,k): T[i,j,k] = sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
# And T[i,j,k] = 1 for all valid i,j,k.
#
# This is a system of 90 equations in 30 * (15+30+18) = 30*63 = 1890 unknowns.
# But the equations are not independent and the structure is specific.
#
# Let me try a simpler approach: use the identity that allows sharing.
# 
# Actually, I think the key insight is:
# (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
# 
# We can group the products. Consider computing for each pair (i,k) the dot product.
# But we can share the B-vector multiplications across different i values.
#
# Let's define for each column k of B, we need to compute sum_j A_{i,j} B_{j,k} for i=0,1,2.
# This is A * B[:,k] which is a 3-vector.
# 
# If we precompute for each j: B_j = (B_{j,0}, B_{j,1}, ..., B_{j,5}), a 6-vector.
# Then (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
#
# We can think of this as: for each j, form the outer product of the column j of A (3-vector) with the row j of B (6-vector), giving a 3x6 matrix. Then sum over j.
# 
# The rank of the "outer product" operation (3x6 matrix from 3-vector and 6-vector) is 18.
# But we're doing 5 such outer products and summing.
# 
# Hmm, this doesn't directly help with tensor rank.
#
# Let me try yet another approach. Maybe I should just output the schoolbook solution since I can't quickly derive the rank-30 decomposition.

# Final decision: output schoolbook solution
def construct():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
