"""
Matrix multiplication 2x2 by 2x7 tensor decomposition.

The bilinear map is:
  C[i,k] = sum_j A[i,j] * B[j,k]
  where A is 2x2, B is 2x7, C is 2x7.

Using the trace convention:
  U has shape (n*m, rank) = (4, rank)
  V has shape (m*p, rank) = (14, rank)
  W has shape (p*n, rank) = (14, rank)
  W[k*n + i, r] is the output factor.

The schoolbook decomposition uses rank = n*m*p = 28, where each
triplet (i, j, k) contributes one rank-1 term.

For a 2x2 x 2x7 multiplication, we can exploit the structure:
  C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
  C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

This is essentially two independent 1x2 x 2x7 multiplications.
Each 1x2 x 2x7 multiplication has rank = 1*2*7 = 14 in schoolbook.
Total rank = 28.

However, we can potentially do better by sharing structure. Let's think:
  For each k, we need:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

This is a 2x2 matrix multiplication for each k (treating B[:,k] as a 2x1 vector
and A as a 2x2 matrix, producing C[:,k] as a 2x1 vector).

Actually, the full tensor is:
  T[i, j, k] where the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * (coefficient for C[i,k])

The tensor has entries T[i*m+j, j*p+k, k*n+i] = 1.

For n=2, m=2, p=7, the schoolbook rank is 28.

Let me check if we can reduce the rank. The key observation is that for each
pair of (i,k), the sum over j is independent. So we have 2*7 = 14 output
entries, each depending on 2 input products. The total number of non-zero
entries in the tensor is 2*2*7 = 28.

Since each rank-1 term can only capture one (i,j,k) triple (because the
indices are all coupled), the rank must be at least 28 for this specific
structure. Wait, that's not necessarily true - a rank-1 term could potentially
capture multiple triples if the factors align.

Actually, let me reconsider. A rank-1 term is of the form:
  u_a * v_b * w_c
  where u is a vector of length 4, v of length 14, w of length 14.

The tensor entry T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some
valid i, j, k.

For a rank-1 term to match multiple (i,j,k) triples, we would need:
  u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for multiple (i,j,k).

This is a very restrictive condition. In general, the rank of the matrix
multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n) and at most
n*m*p. For n=2, m=2, p=7, the lower bound is max(4, 14, 14) = 14, and the
upper bound is 28.

Let me try to see if rank < 28 is achievable. Consider the structure:
  For fixed i, we have C[i,k] = sum_j A[i,j] * B[j,k] for k = 0..6.
  This is a 1x2 times 2x7 multiplication, which has rank 14 in schoolbook.

If we could share between the two i values, we might reduce the rank. But the
A indices are different (A[0,j] vs A[1,j]), so the factors u would need to be
different.

I believe the rank is 28 for this specific case, but let me implement the
schoolbook decomposition as it's guaranteed to be correct.
"""

def construct():
    n, m, p = 2, 2, 7
    rank = n * m * p  # 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
