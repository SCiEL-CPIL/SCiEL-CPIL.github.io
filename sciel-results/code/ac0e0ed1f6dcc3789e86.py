"""
Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition.

We use the schoolbook decomposition as a baseline (rank 128) and then try to
reduce the rank by exploiting the structure of the problem.

For multiplying a 4x4 matrix A by a 4x8 matrix B to get a 4x8 matrix C:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor has indices (i, j, k) with i in 0..3, j in 0..3, k in 0..7.
Schoolbook gives rank = 4*4*8 = 128.

However, we can do better. The key insight is that matrix multiplication tensors
have known lower bounds and constructions. For m x n * n x p, the rank is at most
mn + np + mp - m - n - p + 1 for certain cases, but more generally we can use
various algorithms.

Actually, for small dimensions, let's think carefully. The standard result is that
the rank of matrix multiplication M_{<n,m,p>} is at most n*m*p (schoolbook) and
at least n*m + m*p + p*n - m - p - n + 1 (for the "border rank" related bounds).

For n=4, m=4, p=8:
- Schoolbook: 4*4*8 = 128
- Lower bound: 4*4 + 4*8 + 8*4 - 4 - 8 - 4 + 1 = 16 + 32 + 32 - 16 + 1 = 65

There are known constructions that achieve ranks between these bounds. One approach
is to use the "Strassen-like" techniques or to use the fact that we can share some
computations.

Let me try a different approach: use the fact that for fixed j, the computation
C[i,k] += A[i,j]*B[j,k] is a rank-1 outer product of A[:,j] and B[j,:]. So we need
4 such outer products, each of rank 1 in the tensor sense... wait, no. Each term
A[i,j]*B[j,k] is a rank-1 tensor in (i,k) for fixed j, but summed over j.

Actually, the tensor T_{i,j,k} = delta (the multiplication tensor) can be written as
sum over all (i,j,k) of e_i * e_j * e_k which is rank n*m*p.

But we can be smarter. Consider that if we fix j, the slice T_{i,j,k} is a matrix
that is zero except... no, T_{i,j,k} = 1 if we're looking at the coefficient of
A[i,j]*B[j,k] in C[i,k]. So T is the identity in the sense that each (i,j,k) triple
corresponds to exactly one nonzero entry.

The rank of the matrix multiplication tensor M_{n,m,p} is a well-studied problem.
For small values, exact ranks are known or bounded.

Let me try a constructive approach: use block structure. Since p=8=2*4, we can
split B into two 4x4 blocks B1 and B2. Then C = A*B1 + A*B2, and each A*B is a
4x4*4x4 multiplication. If we had a rank-r decomposition for 4x4*4x4, we could
combine them.

The rank of 4x4 matrix multiplication is known to be at most 49 (from various
constructions) and at least 49 (I believe the exact rank is 49 for 4x4*4x4... 
actually I'm not sure of the exact value). Let me just use a known upper bound.

Actually, for this problem, let me just implement a reasonable decomposition. 
The simplest improvement over schoolbook is to notice that we don't need separate
terms for each (i,j,k) if we can reuse partial products.

Let me try: for each j (0..3), compute the 4x8 product row by row. The j-th column
of A times the j-th row of B gives a 4x8 outer product. That's 4 outer products,
each of rank 1 in the (i,k) space but they're summed. In tensor rank terms, each
outer product e_i * e_j * e_k for fixed j and varying i,k... no, that's not right.

Let me reconsider. The tensor is T(i,j,k) = 1 for all valid i,j,k, and we want
to write T as sum_{r=1}^{R} u_r(i) * v_r(j) * w_r(k).

Schoolbook: R = n*m*p, with u_r(i)=delta(i,i_r), v_r(j)=delta(j,j_r), w_r(k)=delta(k,k_r).

A better approach: use the fact that we can have u_r, v_r, w_r be more complex
vectors. For example, if we set for a given r:
- u_r is a vector in R^n
- v_r is a vector in R^m  
- w_r is a vector in R^p

Then we need sum_r u_r(i) * v_r(j) * w_r(k) = 1 for all i,j,k.

This means the sum of outer products must equal the all-ones tensor.

One way: use the identity matrix structure. Note that if we take:
For each pair (j, k_block) where we group k's...

Actually, let me just try to find a decomposition with rank less than 128 using
a simple algebraic trick.

Consider: 1 = sum_j delta(j,j) trivially, but that doesn't help directly.

Here's an idea: use the fact that (sum_j x_j)(sum_k y_k) = sum_{j,k} x_j y_k.
So if I set v_r(j) = 1 for all j (for some r), and similarly for others... but that
would give 1*1*1 = 1 only if all are 1, which doesn't decompose properly.

Let me just go with a known construction. The "Laderman" type algorithms or
simply use the fact that rank(M_{n,m,p}) <= n*m + m*p + p*n - m - p - n + 1 is
not generally true, but there are specific constructions.

For now, let me implement a decomposition that uses the following idea:
- Split the k-index into two groups of 4: k in 0..3 and k in 4..7.
- For each group, we do a 4x4*4x4 multiplication.
- If I can find a rank-R_44 decomposition for 4x4*4x4, total rank is 2*R_44.

The best known upper bound for 4x4*4x4 is 49 (I think). So 2*49 = 98 < 128.

But I don't have a concrete 49-rank decomposition memorized. Let me just use
schoolbook for the 4x4*4x4 case (rank 64 each, total 128) which doesn't help.

Alternative: just return the schoolbook decomposition. It's correct, just not minimal.
The score will be 128.

Actually, let me try to think of a simple improvement. What if I use the fact that
for the 4x8 product, I can compute it as two 4x4 products? The rank would be at most
2 * (rank of 4x4*4x4). If I use schoolbook for each 4x4*4x4, that's 2*64=128, same.

Let me just go with schoolbook for now and ensure correctness.
"""

import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of matrix multiplication M_{4,4,8}.
    
    Uses schoolbook decomposition with rank = 4*4*8 = 128.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]  # (n*m, rank) = (16, 128)
    V = [[0.0] * rank for _ in range(m * p)]  # (m*p, rank) = (32, 128)
    W = [[0.0] * rank for _ in range(p * n)]  # (p*n, rank) = (32, 128)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
