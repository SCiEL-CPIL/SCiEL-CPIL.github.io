"""
Tensor decomposition for matrix multiplication C = A @ B where:
  A is n x m (4x4)
  B is m x p (4x8)
  C is n x p (4x8)

Output factor W follows the official trace convention W[k*n+i, r]
where k is the column index of C (0..p-1) and i is the row index of C (0..n-1).

The tensor T[i, j, k] = 1 if and only if the (i,j,k) entry of the matrix
multiplication tensor is nonzero. For standard matrix multiplication:
  C[i,k] = sum_j A[i,j] * B[j,k]

So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

We need rank-minimal decomposition:
  T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Note the indexing:
  U has shape (n*m, rank), indexed by (i, j) -> row i*m+j
  V has shape (m*p, rank), indexed by (j, k) -> row j*p+k
  W has shape (p*n, rank), indexed by (k, i) -> row k*n+i  (trace convention)

For schoolbook: rank = n*m*p = 4*4*8 = 128.

We can try to find a lower-rank decomposition. For matrix multiplication of
(n x m) by (m x p), the known lower bounds and constructions vary.

For 4x4 times 4x8, let's think about what's achievable.

One approach: use the fact that we can share factors across different k values
when the structure allows. But for arbitrary matrix multiplication, the tensor
is the standard matmul tensor, and the minimal rank is not known to be less than
n*m*p in general for these small sizes, but there are known improvements for
certain sizes.

Actually, for the standard matrix multiplication tensor M_{n,m,p}, the rank is
known to be at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p) = max(16, 32, 32) = 32.

There are known results: for multiplying an n x n matrix by an n x n matrix,
Strassen's algorithm gives rank 7 for n=2, and various algorithms give better
bounds. For rectangular matrices, there are also improvements.

However, finding the exact minimal rank is hard. Let's try a few approaches:

1. Schoolbook: rank = 128
2. Try to use symmetry / sharing to reduce rank.

Actually, let me reconsider. The tensor for 4x4 * 4x8 multiplication:
  T[i,j,k] = 1 for all i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}

This is a "completely" filled tensor - every entry is 1. This is the tensor
1_{4x4x8}. The rank of the all-ones tensor of size a x b x c is max(a,b,c)
when... wait, no. The rank of the all-ones tensor J_{a,b,c} (where every entry
is 1) is actually... let me think.

J_{a,b,c}[i,j,k] = 1 for all i,j,k.

We can write J_{a,b,c} = sum_{r=1}^{R} u_r v_r w_r where:
  u_r[i] v_r[j] w_r[k] sums to 1 for all i,j,k.

If we set R = 1: u[1]=1, v[1]=1, w[1]=1, then sum = 1. So rank 1 works for
the all-ones tensor!

Wait, that's correct. The all-ones tensor has rank 1 because:
  J[i,j,k] = 1 = 1*1*1

So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1.

Let me verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1 = T[i,j,k].

Yes! The rank is 1.

But wait - is the matrix multiplication tensor really the all-ones tensor? Let me re-read.

The matrix multiplication tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]. For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k. Yes, it is the all-ones tensor.

So the minimal rank is 1.
"""

import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the all-ones tensor, which has rank 1.
    # 
    # Decomposition: T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1
    #
    # U has shape (n*m, 1), V has shape (m*p, 1), W has shape (p*n, 1)
    
    rank = 1
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
