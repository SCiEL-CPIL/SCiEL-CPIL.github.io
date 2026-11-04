"""
Matrix multiplication tensor decomposition for 2x4 * 4x7.

Exploits the n=2 structure: C[1,k] = C[0,k] + D @ B where D = A[1,:] - A[0,:].
This allows us to share the computation of A[0,:] @ B between both output rows.

The key insight:
- C[0,k] = sum_j A[0,j] * B[j,k]  (28 schoolbook terms)
- C[1,k] = sum_j A[1,j] * B[j,k] = sum_j (A[0,j] + D[j]) * B[j,k] = C[0,k] + sum_j D[j] * B[j,k]

So we need:
- 28 terms for A[0,:] @ B (each term: A[0,j]*B[j,k])
- 28 terms for D @ B (each term: D[j]*B[j,k])
- But the "addition" C[1,k] = C[0,k] + (D@B)[k] can be folded into the W factors.

Total rank: 28 + 28 = 56... same as schoolbook. Not helpful directly.

Better approach: Use the fact that for n=2, we can combine terms.
For each j and k, we have two products: A[0,j]*B[j,k] and A[1,j]*B[j,k].
We can write:
  A[0,j]*B[j,k] + A[1,j]*B[j,k] = (A[0,j]+A[1,j])*B[j,k]
  A[0,j]*B[j,k] - A[1,j]*B[j,k] = (A[0,j]-A[1,j])*B[j,k]

So:
  C[0,k] = 1/2 * [(A[0,j]+A[1,j])*B[j,k] + (A[0,j]-A[1,j))*B[j,k]]
  C[1,k] = 1/2 * [(A[0,j]+A[1,j])*B[j,k] - (A[0,j)-A[1,j])*B[j,k]]

This doesn't reduce the number of bilinear multiplications (still 2 per j,k pair),
but it shows the structure.

Let me try a different approach: use the known rank for [2,4,7].
The schoolbook rank is 56. For small matrices, there may not be a known improvement.

However, we can try to construct a valid decomposition with rank < 56 by exploiting
the specific dimensions. Let me think about this more carefully.

For [2,4,7]:
- 8 A-entries, 28 B-entries, 14 C-entries
- Schoolbook: 56 rank-1 terms

One approach: group the j-index. For each j (0..3), we have a 2x1 times 1x7 multiplication,
which has rank 2*7 = 14 (schoolbook) or potentially less.

Actually, for a 2x1 times 1x7 matrix multiplication, the tensor rank is exactly 2*7 = 14
in the schoolbook, but can it be less? The output is a 2x7 matrix, each entry is a product
of one A-entry and one B-entry. So we need 14 independent products. The rank is 14.

So for 4 such groups (j=0,1,2,3), total rank is 4*14 = 56. Same as schoolbook.

To improve, we need to share work across j-values. This is where more sophisticated
algorithms come in, but for these small dimensions, the improvement may be limited.

Let me just implement the schoolbook decomposition correctly and verify it works.
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
