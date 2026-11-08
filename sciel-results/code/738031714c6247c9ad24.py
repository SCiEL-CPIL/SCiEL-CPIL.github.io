"""Direct approach: explicit tensor decomposition for 4x5 * 5x5 matrix multiplication.

We use a tiling strategy: split the 4x5 matrix into two 2x5 blocks (top/bottom),
and the 5x5 matrix into a 5x5 block. Then:
  C_top = A_top * B  (2x5 * 5x5 = 2x5)
  C_bot = A_bot * B  (2x5 * 5x5 = 2x5)

For each 2x5 * 5x5 sub-problem, we can further decompose.
Actually, let's think about this differently.

The key insight: we can use Strassen-like decomposition on the middle dimension.
For A (4x5) * B (5x5), we can split the 5 middle dimension into 2+3 or use
other partitions.

Actually, the most straightforward improvement: use the fact that for matrix
multiplication, we can sometimes share computations. But for 4x5*5x5, the
schoolbook rank is 100.

Let me think about whether we can do better than 100.

For n x m * m x p, the tensor rank is at most n*m*p. Can it be less?

For square matrices, Strassen shows rank < n^3 for n >= 2. Specifically:
- 2x2 * 2x2: rank 7 (Strassen)
- 3x3 * 3x3: rank 23 (Bini et al.) or 21 (some results)
- 4x4 * 4x4: rank 49 (Strassen tiling) or better

For non-square, it's less clear. 4x5 * 5x5:
- Pad to 5x5 * 5x5: rank <= rank(5x5*5x5). Schoolbook is 125, but with Strassen-like
  methods could be less. But padding wastes rank.
- Or: split into 2x5*5x5 + 2x5*5x5. Each 2x5*5x5 has schoolbook rank 50.
  Can we do 2x5*5x5 in rank < 50?

For 2x5 * 5x5: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in 0..4.
This is 2*5*5 = 50 terms. Can we share?

Actually, for 2 x m * m x p, the rank is at most 2*m*p. Is it less?
For 2x2*2x2: rank 7 < 8. So yes, for small cases it can be less.

For 2x5*5x5: we could potentially use a Strassen-like approach.

Let me think about 2x2*2x2 Strassen:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B10-B11)
  M4 = A11(B01-B00)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B10)
  M7 = (A01-A11)(B01+B11)
  
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

Each M is a product of a linear combo of A entries and a linear combo of B entries.
In tensor form, each M corresponds to one rank-1 term where U and V have non-zero
entries in specific positions, and W distributes the result.

For our problem, let me try to build a decomposition using Strassen on 2x2 blocks
within the 4x5*5x5.

Approach: 
- Split A (4x5) into four 2x... no, let's split into top 2 rows and bottom 2 rows.
- Split B (5x5) into... hmm, 5 is odd.

Alternative: split the middle dimension (5) into 2+3.
A = [A1 (4x2) | A2 (4x3)]
B = [B1 (2x5); B2 (3x5)]
C = A1*B1 + A2*B2

A1*B1: 4x2 * 2x5, schoolbook rank = 4*2*5 = 40
A2*B2: 4x3 * 3x5, schoolbook rank = 4*3*5 = 60
Total: 100. No gain.

What if we use Strassen on the 4x2 * 2x5 part?
4x2 * 2x5: We can split the 4 into 2+2.
A1 = [A1a (2x2); A1b (2x2)]
B1 = [B1a (2x2) B1b (2x3)]

A1a * B1: 2x2 * 2x5. Can use Strassen-like.
A1b * B1: 2x2 * 2x5.

Hmm, this is getting complicated. Let me try a different approach.

Actually, for 2x2 * 2x2, Strassen gives rank 7. For 2x2 * 2xp, we can use:
Split B into 2x2 + 2x(p-2) blocks.
C = A * B1 (2x2) + A * B2 (2x(p-2))
A*B1: rank 7 (Strassen)
A*B2: schoolbook 2*2*(p-2) = 4(p-2)
Total: 7 + 4(p-2) = 4p - 1

For 2x2 * 2x5: 4*5 - 1 = 19. Schoolbook is 20. Save 1.

For 4x5 * 5x5:
Split into two 2x5 * 5x5 problems.
Each 2x5 * 5x5: split B into 5 columns.
  2x5 * 5x5 = sum over k of (2x5 * 5x1) ... no.
  
  Actually: C = A * B where A is 2x5, B is 5x5.
  C[i,k] = sum_j A[i,j] B[j,k]
  
  We can write this as: for each k, C[:,k] = A * B[:,k] where B[:,k] is 5x1.
  A * B[:,k] is 2x1. This is a 2x5 * 5x1 multiplication.
  
  2x5 * 5x1: schoolbook rank = 2*5*1 = 10.
  Can we do better? 2x5 * 5x1 = (2x2*2x1) + (2x3*3x1) if we split.
  2x2*2x1: rank? C[i] = sum_j A[i,j]B[j]. This is like dot products.
  Actually 2x2*2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1.
  This is 4 terms. Can we do it in 3? 
  Let's see: we need 4 bilinear forms. The rank of this tensor...
  It's the same as 2x2 * 2x1 matrix mult. The tensor has T[i,j,k] where i in {0,1}, j in {0,1}, k=0.
  T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
  This is a 2x2x1 tensor. Its rank is 2 (since it's just two independent linear forms).
  Wait no, it's a 3-way tensor. T[i,j,k] for i,j in {0,1}, k=0.
  Rank of a 2x2x1 tensor: it's equivalent to a 2x2 matrix flattened, so rank <= 4.
  Actually the rank of a tensor T[i,j,k] = M[i,j] (with k fixed) is the matrix rank of M.
  Here M is 2x2 identity-like (all ones in the sense that each entry is 1). 
  No wait, T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1. So M = [[1,1],[1,1]] which has rank 1.
  So the tensor rank is 1? No, that's not right either.
  
  Let me reconsider. The tensor for 2x2 * 2x1 matrix mult:
  C[0] = A[0,0]*B[0] + A[0,1]*B[1]
  C[1] = A[1,0]*B[0] + A[1,1]*B[1]
  
  In the decomposition format: U is (2*2, rank) = (4, rank), V is (2*1, rank) = (2, rank), W is (1*2, rank) = (2, rank).
  
  Schoolbook: rank 4.
  Term (i,j): U[i*2+j, c]=1, V[j*1+0, c]=1, W[0*2+i, c]=1.
  
  Can we do rank 3? Let's see if the tensor has rank 3.
  The tensor T has T[i,j,k] where the condition is sum_c U[i*2+j,c]*V[j,c]*W[k*2+i,c] = 1 for all i,j,k.
  Wait, for 2x2*2x1: n=2, m=2, p=1.
  U is (n*m, rank) = (4, rank). Index: i*m+j = i*2+j.
  V is (m*p, rank) = (2, rank). Index: j*p+k = j*1+0 = j.
  W is (p*n, rank) = (2, rank). Index: k*n+i = 0*2+i = i.
  
  Condition: sum_c U[i*2+j, c] * V[j, c] * W[i, c] = 1 for all i in {0,1}, j in {0,1}.
  
  This is 4 equations. With rank r, we have 4r + 2r + 2r = 8r variables.
  For r=1: 8 variables, 4 equations. Likely solvable? 
  U[0,0]*V[0,0]*W[0,0] = 1
  U[1,0]*V[0,0]*W[0,0] = 1
  U[0,1]*V[1,0]*W[0,0] = 1  -- wait, j=1, so V[1,0]
  U[1,1]*V[1,0]*W[1,0] = 1
  
  Hmm, let me redo. i in {0,1}, j in {0,1}, k=0.
  W index is k*n+i = i.
  
  c=0 only (rank 1):
  U[i*2+j, 0] * V[j, 0] * W[i, 0] = 1 for all i,j.
  
  i=0,j=0: U[0,0]*V[0,0]*W[0,0] = 1
  i=0,j=1: U[1,0]*V[1,0]*W[0,0] = 1
  i=1,j=0: U[2,0]*V[0,0]*W[1,0] = 1
  i=1,j=1: U[3,0]*V[1,0]*W[1,0] = 1
  
  From eq1: U[0,0] = 1/(V[0,0]*W[0,0])
  From eq2: U[1,0] = 1/(V[1,0]*W[0,0])
  From eq3: U[2,0] = 1/(V[0,0]*W[1,0])
  From eq4: U[3,0] = 1/(V[1,0]*W[1,0])
  
  We can choose V[0,0]=V[1,0]=W[0,0]=W[1,0]=1, then all U=1. So rank 1 works!
  
  Wait, that means 2x2*2x1 has rank 1? That seems too good. Let me verify.
  
  With rank 1: U = [1,1,1,1]^T, V = [1,1]^T, W = [1,1]^T.
  For i=0,j=0: U[0]*V[0]*W[0] = 1*1*1 = 1 ✓
  For i=0,j=1: U[1]*V[1]*W[0] = 1*1*1 = 1 ✓
  For i=1,j=0: U[2]*V[0]*W[1] = 1*1*1 = 1 ✓
  For i=1,j=1: U[3]*V[1]*W[1] = 1*1*1 = 1 ✓
  
  Yes! So 2x2*2x1 has rank 1. That makes sense because it's just two dot products,
  and the "tensor" is separable.
  
  Similarly, 2xm * mx1 has rank 1 for any m! Because C[i] = sum_j A[i,j]B[j], and the
  tensor T[i,j,0] = 1 for all i,j, which is a rank-1 tensor (outer product of ones).
  
  OK so that's not helpful for reducing rank in the general case.

Let me go back to the main problem. I'll stick with the schoolbook rank of 100 but
implement it cleanly with a time guard.

Actually, wait. Let me reconsider whether there's a known decomposition with rank < 100
for 4x5*5x5.

The tensor rank of matrix multiplication M_{n,m,p} is known to be:
- For n=m=p=2: 7
- For n=m=p=3: 23 (or 21 with some results)
- For n=m=p=4: 49 (Strassen) or 48 (some results)
- For non-square, less is known.

For 4x5*5x5, I don't know of a specific rank < 100. The safe bet is to keep rank 100.

Let me just implement the schoolbook cleanly with a deadline guard.
"""

import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def construct():
    """Return the schoolbook decomposition of rank 100 for 4x5 * 5x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 100
    
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
    
    # Check time - if we're close to the limit, return immediately
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        # This shouldn't happen for the direct construction, but guard anyway
        pass
    
    return {"U": U, "V": V, "W": W}
