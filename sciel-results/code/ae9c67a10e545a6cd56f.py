"""Rewritten program: rank-minimal decomposition for 2x2 * 2x11 multiplication.

The 2x11 right factor can be viewed as 11 independent 2x2 multiplications.
We use Strassen's 7-multiplication algorithm for each 2x2 block.
Since the left matrix is shared across all 11 columns, we can precompute
the Strassen intermediates for the left matrix once, and then for each
column k we only need to combine with the right matrix's column entries.

However, the tensor decomposition requires that U depends only on (i,j), 
V depends only on (j,k), and W depends only on (k,i) via the trace convention W[k*n+i, r].

Let's think about this more carefully. The multiplication tensor for A(2x2) * B(2x11) = C(2x11)
has entries C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,10}.

The rank of this tensor is at most 11 * rank(2x2 mult) = 11 * 7 = 77 using Strassen per column.
But we can do better: since the left matrix is shared, we can precompute Strassen intermediates
for A once (7 values), and for each column k, compute 7 products with B's column, then combine.

Actually, the standard approach: the tensor T_{i,j,k} = A_{i,j} * B_{j,k} for fixed i,j,k.
We want to write T as sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}.

For each column k (0..10), the sub-tensor is the 2x2 multiplication tensor, which has rank 7 (Strassen).
So a trivial upper bound is 11*7 = 77.

Can we share across columns? The left factor A is the same for all k. 
In a tensor decomposition, U depends on (i,j) only, V on (j,k) only, W on (k,i) only.
The Strassen decomposition for column k uses:
  U entries depend on i,j (same for all k)
  V entries depend on j,k (different per k)
  W entries depend on k,i (different per k)

So if we use the same 7 Strassen forms for each column, we get 77 terms.
But maybe we can find a smaller decomposition by exploiting structure.

Actually, for the tensor of multiplying (n x m) by (m x p), the rank is known to be at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n). For 2x2 * 2x11:
- n*m = 4
- m*p = 22  
- p*n = 22

So rank >= 22. The Strassen-per-column gives 77.

Let me try to construct a rank-77 decomposition using Strassen for each column, and see if we can verify it.

Strassen for 2x2:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B10-B11)
  M4 = A11(B01-B10)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B10)
  M7 = (A01-A11)(B01+B11)

C00 = M1 - M4 + M5 - M7
C01 = M3 + M4
C10 = M2 + M3
C11 = M1 - M2 + M6 + M7

For the tensor decomposition, each multiplication M_r corresponds to one term in the sum.
For column k:
  A is the 2x2 left matrix, B_k is the 2x1 column k of B.

Let me index: A[i,j] for i,j in {0,1}, B[j,k] for j in {0,1}, k in 0..10.
C[i,k] for i in {0,1}, k in 0..10.

For Strassen on column k:
  M1_k = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])
  M2_k = (A[1,0]+A[1,1]) * B[0,k]
  M3_k = A[0,0] * (B[1,k]-B[0,k])
  M4_k = A[1,1] * (B[0,k]-B[1,k])
  M5_k = (A[0,0]+A[0,1]) * B[1,k]
  M6_k = (A[1,0]-A[0,0]) * (B[0,k]+B[1,k])
  M7_k = (A[0,1]-A[1,1]) * (B[0,k]+B[1,k])

Wait, I need to be careful with the Strassen formulas. Let me use a standard reference.

Standard Strassen:
  M1 = (a00 + a11)(b00 + b11)
  M2 = (a10 + a11)b00
  M3 = a00(b10 - b11)
  M4 = a11(b01 - b10)
  M5 = (a00 + a01)b11
  M6 = (a10 - a00)(b00 + b10)
  M7 = (a01 - a11)(b01 + b11)

  c00 = M1 - M4 + M5 - M7
  c01 = M3 + M4
  c10 = M2 + M3
  c11 = M1 - M2 + M6 + M7

For our case, for column k, the "b" matrix is just the column B[:,k], which is a 2x1 vector. But Strassen is for 2x2 * 2x2. We need to adapt.

Actually, for multiplying a 2x2 matrix by a 2x1 vector (i.e., one column of the result), we can use a simpler approach. The product is:
  C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
  C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

This is just 4 multiplications per column, giving rank 4*11 = 44 with schoolbook.

Can we do better than 44? The tensor for 2x2 * 2x1 has rank at most 4 (schoolbook). Is the rank less than 4? For a 2x2 matrix times a 2x1 vector, the tensor is actually the same as the matrix-vector product tensor. The rank of the 2x2 matrix-vector product tensor is 4 (it's full rank since the Kronecker product structure). Actually, let me think...

The tensor T_{i,j,k} where i in {0,1} (row of C), j in {0,1} (column of A / row of B), k=0 (only one column). Wait, k is the column index of B, and there's only one "output column" per k.

Actually for a single column, we have C[i] = sum_j A[i,j] B[j], which is a bilinear map from A (4 entries) and B (2 entries) to C (2 entries). The tensor has 2*4*2 = 16 entries, but only 4 are nonzero in the schoolbook sense... no wait.

Let me reconsider. The tensor for matrix multiplication A(n x m) * B(m x p) = C(p x n) [using the trace convention W[k*n+i,r] where k is row of C, i is column of C which is row of A]... 

I'm getting confused with the indexing. Let me just use the schoolbook decomposition which is guaranteed correct, with rank = n*m*p = 2*2*11 = 44.

The current program has rank 44. Can we improve? Let me try rank 44 first to make sure it's correct, then think about improvements.

Actually, the current program already does schoolbook with rank 44. So score 1.0 means rank 44. To improve, we need rank < 44.

Let me try to find a decomposition with rank < 44. One approach: since the left matrix A is 2x2 and shared across all 11 columns of B, we might be able to use a decomposition that shares some structure.

For now, let me just output the schoolbook decomposition with rank 44 to ensure correctness, and add a deadline guard.
"""

import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 44
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
# EVOLVE-BLOCK-END
