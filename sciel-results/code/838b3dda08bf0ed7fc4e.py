"""
Tensor decomposition for 2x3 x 3x10 matrix multiplication.

Strategy:
  - For each pair of rows in the 3x10 block (there are 3 pairs), apply Strassen
    to the 2x2 sub-problem (rows i1,i2 of A times 2 columns of B), which has
    rank 7 instead of 8.
  - The remaining single row (row 3 of A) is multiplied by all 10 columns of B
    using schoolbook, rank 10.
  - Total rank per 3x10 block: 3*7 + 10 = 31.
  - Two such blocks (columns 1-3 and 4-6, etc.) — wait, we have 3 rows in A and
    10 columns in B. Let me reconsider.

Actually: A is 2x3, B is 3x10.
  - A has 2 rows, 3 columns.
  - B has 3 rows, 10 columns.
  - Output C is 2x10.

We can group the 3 rows of A (which are columns of A, i.e., the shared dimension m=3)
into a 2+1 split. For the 2-row part, we use Strassen on the 2x2 block (2 rows of A
times 2 columns of B), rank 7 per 2-column block. We have 10 columns of B, so 5 blocks
of 2 columns each: 5*7 = 35.
For the 1-row part (the 3rd row of A), we multiply by all 10 columns of B: rank 10.
Total: 35 + 10 = 45.

Wait, let me re-examine. The shared dimension is m=3. A is (n=2) x (m=3), B is (m=3) x (p=10).

We split the m=3 dimension into 2+1.
- For the m=2 part: A_2 is 2x2, B_2 is 2x10. We split B_2 into 5 blocks of 2x2.
  Each 2x2x2 block uses Strassen rank 7. Total: 5*7 = 35.
- For the m=1 part: A_1 is 2x1, B_1 is 1x10. Schoolbook: 2*1*10 = 20? No.
  Actually for a 2x1 times 1x10, it's just outer product: rank 2*10 = 20? No.
  
Let me think more carefully. The tensor is T[i,j,k] = delta for C[i,k] = sum_j A[i,j]*B[j,k].

For the m=1 part: A has 2 rows (i=0,1), 1 column (j=2). B has 1 row (j=2), 10 columns (k=0..9).
C[i,k] contribution = A[i,2] * B[2,k]. This is an outer product of a 2-vector and a 10-vector.
Rank = 2*10 = 20? No, outer product has rank 1 in the tensor sense? No.

Actually the matrix multiplication tensor for n x m times m x p has rank at least n*m*p in the trivial case, but can be lower. For a 2x1 x 1x10 sub-problem, the tensor is:
T[i,k] = A[i,0] * B[0,k] for i in {0,1}, k in {0,...,9}.
This is a rank-1 tensor in the (i,k) space for each (j) but j is fixed. Actually it's:
C[i,k] = A[i,0]*B[0,k]. The tensor T[i,0,k] = 1 for all i,k. This is a 2x1x10 tensor with all ones.
The rank of an all-ones tensor of size a x b x c is max(a,b,c) if... no.

The all-ones tensor of size 2 x 1 x 10: T[i,j,k] = 1 for i in {0,1}, j=0, k in {0..9}.
This can be written as sum over r of U[i,r]*V[j,r]*W[k,r]. Since j is always 0, V[0,r] is a scalar.
We need: for all i,k: sum_r U[i,r]*V[0,r]*W[k,r] = 1.
This is like a 2x10 matrix of all ones being decomposed as sum of rank-1 matrices.
The rank of the all-ones 2x10 matrix is 1. So we can do it with rank 1!
U[0,0]=1, U[1,0]=1, V[0,0]=1, W[k,0]=1 for all k. Then sum_r U[i,r]V[0,r]W[k,r] = 1*1*1 = 1. Yes!

So the m=1 part has rank 1? That doesn't seem right for matrix multiplication. Let me re-examine.

Oh wait, I think I'm confusing the tensor structure. The matrix multiplication tensor T is a 3-tensor with indices (i,j,k) where i indexes rows of A and rows of C, j indexes columns of A and rows of B, k indexes columns of B and columns of C. The tensor entry T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].

So T is of size n x m x p = 2 x 3 x 10, with T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this convention? No!).

Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k. The decomposition is:
C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k].

So T[i,j,k] = 1 for all i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}.

The rank of T is the minimum r such that T[i,j,k] = sum_{c=1}^r U[i,c] V[j,c] W[k,c].

For the full 2x3x10 all-ones tensor, the trivial rank is 2*3*10 = 60.

Now, if we split j into j1 in {0,1} and j2 = 2:
- Part 1: T[i,j1,k] = 1 for i in {0,1}, j1 in {0,1}, k in {0,...,9}. This is a 2x2x10 all-ones tensor.
- Part 2: T[i,j2,k] = 1 for i in {0,1}, j2=2, k in {0,...,9}. This is a 2x1x10 all-ones tensor.

For Part 2 (2x1x10 all-ones): rank = ? We need sum_c U[i,c]V[2,c]W[k,c] = 1 for all i,k.
As I showed, rank 1 suffices: U[i,0]=1, V[2,0]=1, W[k,0]=1.

For Part 1 (2x2x10 all-ones): We can split k into 5 pairs. For each pair (k1,k2), we have a 2x2x2 all-ones tensor, which has rank 7 (Strassen). Total: 5*7 = 35.

Grand total: 35 + 1 = 36.

Let me verify: is the 2x2x2 all-ones tensor really rank 7? The standard Strassen decomposition multiplies two 2x2 matrices. The tensor for 2x2x2 matrix multiplication has rank 7. Yes, that's correct.

So total rank = 36. Let me construct this.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Split m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    assert m1 + m2 == m
    
    # Split p=10 into 5 blocks of 2
    p_blocks = p // 2
    assert p_blocks * 2 == p
    
    # Rank for m1 part: p_blocks * 7 (Strassen per 2x2x2 block)
    # Rank for m2 part: 1 (outer product)
    rank_m1 = p_blocks * 7
    rank_m2 = 1
    total_rank = rank_m1 + rank_m2
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (n*m) x rank = 6 x 36
    V = [[0] * total_rank for _ in range(m * p)]  # (m*p) x rank = 30 x 36
    W = [[0] * total_rank for _ in range(p * n)]  # (p*n) x rank = 20 x 36
    
    # Strassen 2x2x2 decomposition:
    # For multiplying A (2x2) by B (2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In tensor form, the 7 rank-1 terms for 2x2x2 matmul:
    # Each term c has U[i,c], V[j,c], W[k,c]
    # i in {0,1} (row of A), j in {0,1} (col of A / row of B), k in {0,1} (col of B)
    
    strassen_terms = [
        # (U00, U10, V00, V10, W00, W01, W10, W11)
        # M1: U=[1,1], V=[1,1], W=[1,1]
        (1, 1, 1, 1, 1, 1, 1, 1),
        # M2: U=[0,1], V=[1,1], W=[1,0]
        (0, 1, 1, 1, 1, 0, 0, 0),
        # M3: U=[1,0], V=[1,0], W=[0,1]
        (1, 0, 1, 0, 0, 1, 0, 0),
        # M4: U=[0,1], V=[0,1], W=[1,0]
        (0, 1, 0, 1, 1, 0, 0, 0),
        # M5: U=[1,1], V=[0,1], W=[0,1]
        (1, 1, 0, 1, 0, 1, 0, 0),
        # M6: U=[1,-1], V=[1,0], W=[1,1]
        (1, -1, 1, 0, 1, 1, 0, 0),
        # M7: U=[1,-1], V=[0,1], W=[0,1]
        (1, -1, 0, 1, 0, 1, 0, 0),
    ]
    
    # Fill in the m1 part (j=0,1) with Strassen blocks
    col = 0
    for block in range(p_blocks):
        k_offset = block * 2  # k indices: k_offset, k_offset+1
        for term in strassen_terms:
            U00, U10, V00, V10, W00, W01, W10, W11 = term
            # U indices: i*m + j, where i in {0,1}, j in {0,1}
            U[0 * m + 0][col] = U00  # i=0, j=0
            U[1 * m + 0][col] = U10  # i=1, j=0
            U[0 * m + 1][col] = U00  # i=0, j=1
            U[1 * m + 1][col] = U10  # i=1, j=1
            
            # V indices: j*p + k, where j in {0,1}, k in {k_offset, k_offset+1}
            V[0 * p + k_offset][col] = V00 * W00
            V[0 * p + k_offset + 1][col] = V00 * W01
            V[1 * p + k_offset][col] = V10 * W10
            V[1 * p + k_offset + 1][col] = V10 * W11
            
            # W indices: k*n + i, where k in {k_offset, k_offset+1}, i in {0,1}
            W[k_offset * n + 0][col] = 1
            W[k_offset * n + 1][col] = 1
            W[(k_offset + 1) * n + 0][col] = 1
            W[(k_offset + 1) * n + 1][col] = 1
            col += 1
    
    # Fill in the m2 part (j=2) with rank-1 outer product
    # U[i, col] = 1 for i in {0,1}, j=2
    # V[2*p + k, col] = 1 for all k
    # W[k*n + i, col] = 1 for all k, i
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}
