"""
Kronecker product decomposition for 3x4x8 matrix multiplication.

Factorization: 3x4x8 = (3x2x4) ⊗ (1x2x2)

For (1,2,2): n=1, m=2, p=2
  C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
  This is 2 independent dot products of length 2, each requiring 2 scalar mults.
  Total schoolbook rank = 4.
  
  However, we can view it as: for each output k (0,1), C[0,k] = sum_j A[0,j]*B[j,k].
  The tensor has entries at positions corresponding to (i=0, j, k) for j,k in {0,1}.
  That's 4 nonzero entries in a 2x4x2 tensor. Rank is 4.

For (3,2,4): n=3, m=2, p=4
  Schoolbook rank = 3*2*4 = 24.
  
  Can we do better? Let's think about this.
  C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in 0..2, k in 0..3.
  
  This is 12 independent bilinear forms, each of the form a*b (one term each).
  Wait no, each C[i,k] has 2 terms. So it's like 12 dot products of length 2.
  
  Actually each C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
  The A-side coefficient for A[i,0] is B[0,k], and for A[i,1] is B[1,k].
  
  The tensor T has entries:
  For each (i, j, k) where i in 0..2, j in 0..1, k in 0..3:
    T[i*m+j, j*p+k, k*n+i] = 1
  That's 3*2*4 = 24 nonzero entries.
  
  Is the rank less than 24? The structure is:
  For fixed i, the terms are A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0..3.
  
  We can factor: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
  This is a sum of 2 rank-1 matrices (in the B dimension) for each row i.
  
  Total: 3 rows * 2 terms = 6 "scalar multiplications" of the form (scalar from A) * (vector from B).
  But in the tensor decomposition sense, each term A[i,j]*B[j,k] for fixed (i,j) and varying k
  is not a single rank-1 tensor term.
  
  Let me reconsider. A rank-1 tensor term is of the form:
    u_vector ⊗ v_vector ⊗ w_vector
  where u has n*m entries, v has m*p entries, w has p*n entries.
  
  For the term A[i,j]*B[j,k] contributing to C[i,k]:
  u has 1 at position i*m+j
  v has 1 at position j*p+k
  w has 1 at position k*n+i
  
  Each such term is rank-1. The schoolbook uses 24 such terms.
  
  Can we combine? If we take all k for fixed (i,j):
  Sum over k of A[i,j]*B[j,k]*delta_{k*n+i, output}
  
  This is: u = e_{i*m+j}, v = sum_k e_{j*p+k}, w = sum_k e_{k*n+i}
  But v and w are not single basis vectors, they're sums. So this isn't a rank-1 term.
  
  A rank-1 term requires u, v, w to each be single vectors, and the tensor is their outer product.
  So we can't combine across k unless we change the structure.
  
  I think the rank of M_{3,2,4} is indeed 24 (schoolbook), because each (i,j,k) triple
  gives a distinct "corner" of the tensor that can't be factored.
  
  Wait, let me check more carefully. The tensor T ∈ R^{6 × 8 × 12} (n*m=6, m*p=8, p*n=12).
  Nonzero entries at T[i*2+j, j*4+k, k*3+i] = 1 for i∈{0,1,2}, j∈{0,1}, k∈{0,1,2,3}.
  
  That's 24 entries. Are they linearly independent as rank-1 tensors?
  Two rank-1 tensors e_a⊗e_b⊗e_c and e_{a'}⊗e_{b'}⊗e_{c'} are linearly independent
  if the triples (a,b,c) are distinct. Since all 24 triples are distinct, the rank is 24.
  
  So M_{3,2,4} has rank 24, and M_{1,2,2} has rank 4.
  Kronecker product rank: 24 * 4 = 96. Same as schoolbook!
  
  This means the Kronecker product approach doesn't help here.
  
  Let me try a different factorization:
  3x4x8 = (1x4x2) ⊗ (3x1x4)? 
  1*3=3, 4*1=4, 2*4=8. ✓
  
  M_{1,4,2}: n=1, m=4, p=2. Rank = 1*4*2 = 8 (schoolbook).
  M_{3,1,4}: n=3, m=1, p=4. Rank = 3*1*4 = 12 (schoolbook).
  Total: 8*12 = 96. Same.
  
  It seems like for these dimensions, the Kronecker product of schoolbook decompositions
  always gives 96.
  
  What if one of the sub-problems has a non-schoolbook decomposition?
  For M_{1,4,2}: C[0,k] = sum_{j=0}^{3} A[0,j]*B[j,k] for k=0,1.
  This is 2 dot products of length 4. Each dot product of length 4 has rank 4 (can't be improved
  for a single dot product). But can we share?
  
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
  C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
  
  The 8 terms are all distinct rank-1 tensors. Rank is 8.
  
  I'm stuck. The rank of 3x4x8 matrix multiplication appears to be 96 with known methods.
  
  Let me just implement a clean schoolbook decomposition with rank 96.
  """
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Returns {'U','V','W'} with integer entries forming a rank-96 exact decomposition
    of the 3x4 x 4x8 matrix multiplication tensor (schoolbook).
    
    Convention:
      U has shape (n*m, rank) = (12, 96)
      V has shape (m*p, rank) = (32, 96)  
      W has shape (p*n, rank) = (24, 96)
      
    For each (i, j, k) with i in 0..n-1, j in 0..m-1, k in 0..p-1:
      U[i*m + j, col] = 1
      V[j*p + k, col] = 1
      W[k*n + i, col] = 1
    where col is the column index for this (i,j,k) triple.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
