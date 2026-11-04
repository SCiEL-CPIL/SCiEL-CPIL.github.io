"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Uses the standard bilinear tensor decomposition convention:
  C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c]) * W[k*n+i, c]

Schoolbook: one term per (i,j,k) triple, rank = n*m*p = 56.

We attempt to reduce rank by exploiting the n=2 structure:
  C[0,k] = sum_j A[0,j] B[j,k]
  C[1,k] = sum_j A[1,j] B[j,k]

Key idea: For each j, the pair (A[0,j], A[1,j]) contributes to both rows.
We can group terms by j. For a fixed j, the contribution to C is:
  C[:,k] += A[:,j] * B[j,k]

This is an outer product of a 2-vector (A[:,j]) and a 1-value (B[j,k]) producing a 2-vector.
In the tensor decomposition, for each (j,k) we need to produce the 2 outputs C[0,k] and C[1,k].

Standard approach for [2,4,7]:
- For each j (4 choices) and k (7 choices), we have a "column" of A (2 entries) times B[j,k].
- The bilinear form for fixed (j,k) maps (A[:,j], B[j,k]) -> (A[0,j]*B[j,k], A[1,j]*B[j,k]).
- This is a rank-2 bilinear map (in the sense that it produces 2 independent outputs from 2*1 inputs).
- But in our tensor decomposition framework, each "term" c contributes one product of two linear forms times a W vector.

Actually, let me think about this differently. The tensor M has entries M[i,j,k] = delta (it's the identity in the j-index).
The rank of the matmul tensor [n,m,p] is the minimum number of rank-1 terms.

For [2,4,7], known bounds:
- Upper bound: 56 (schoolbook)
- Lower bound: max(2*7, 4*7, 2*4) = max(14,28,8) = 28 (trivial lower bound from dimension)
- Better lower bounds exist but I'll work with what I can construct.

Strategy: Use the fact that for n=2, we can pair up terms.
For each j and k, instead of 2 separate terms (one for i=0, one for i=1), 
we can use 1 term if the W vector can handle both outputs.

Wait - in the decomposition, each term c has:
  U_c: linear form on A (8-dim)
  V_c: linear form on B (28-dim)  
  W_c: vector in output space (14-dim, reshaped to (7,2) or (2,7) depending on convention)

The contribution of term c to C[i,k] is:
  (U_c^T * vec(A)) * (V_c^T * vec(B)) * W_c[k*n+i]

For schoolbook term (i0, j0, k0):
  U_c picks out A[i0, j0]
  V_c picks out B[j0, k0]
  W_c has 1 at position k0*n+i0

So C[i,k] gets contribution A[i0,j0]*B[j0,k0] * delta_{i,i0} * delta_{k,k0}

To reduce rank, we need to find terms where U_c and V_c are "sparser" or where W_c has multiple non-zero entries.

Idea: For a fixed j, consider the 2x7 sub-problem: A[:,j] (2x1) times B[j,:] (1x7) -> 2x7 matrix.
This is just an outer product. In bilinear terms:
  C[i,k] += A[i,j] * B[j,k] for i=0,1 and k=0..6.

Can we represent this outer product with fewer than 14 terms (2*7)?
The outer product of a 2-vector and a 7-vector has tensor rank 2*7 = 14 in general... 
but wait, we're not decomposing the outer product tensor, we're using it as part of the bilinear map.

Actually, the bilinear map (x, y) -> x * y^T (outer product) where x is 2-dim and y is 7-dim
has tensor rank 2*7 = 14 in the standard sense. So no savings there.

Let me try a different approach: exploit linear dependencies.

C[1,k] - C[0,k] = sum_j (A[1,j] - A[0,j]) B[j,k]

Let D[j] = A[1,j] - A[0,j]. Then:
C[1,k] = C[0,k] + sum_j D[j] B[j,k]

So if we compute:
  X[k] = C[0,k] = sum_j A[0,j] B[j,k]  (rank 4*7 = 28 for the [1,4,7] part)
  Y[k] = sum_j D[j] B[j,k]              (rank 4*7 = 28 for the [1,4,7] part on D)

Then C[0,k] = X[k] and C[1,k] = X[k] + Y[k].

Total rank: 28 + 28 = 56. Same as schoolbook! No savings.

The issue is that D depends on A linearly, so the bilinear map for Y is still full rank.

Let me try yet another approach. What if we use Strassen-like tricks?

For 2x2 matrix multiplication, Strassen uses 7 multiplications instead of 8.
Can we apply similar ideas here?

The 2x4 * 4x7 problem can be seen as two 1x4 * 4x7 problems.
Each 1x4 * 4x7 is a dot product of length 4 with each column of B.

Hmm, this is getting complex. Let me just verify the schoolbook is correct and submit it.
The rank is 56, which is the baseline.

Actually, wait. Let me reconsider the problem. The tensor rank of [2,4,7] matmul:
- The trivial upper bound is 56.
- There might be constructions with lower rank.

For [2,m,p], one known construction: use the fact that 2x2 blocks can be multiplied with 7 terms (Strassen).
If we group the 4 columns of A into two pairs and the 7 columns of B into... this doesn't divide evenly.

Let me just go with the schoolbook decomposition of rank 56. It's valid and correct.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
