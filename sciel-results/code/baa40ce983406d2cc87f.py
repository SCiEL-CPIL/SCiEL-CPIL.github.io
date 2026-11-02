"""
Matrix multiplication 2x2 by 2x15: find a rank-minimal exact tensor decomposition.

The multiplication tensor T has entries T[i, j, k] = 1 for the standard schoolbook
product C[i,k] = sum_j A[i,j] B[j,k].  We seek factor matrices
U (n*m x R), V (m*p x R), W (p*n x R) such that for every (i,j,k),

    sum_r  U[i*m + j, r] * V[j*p + k, r]  ==  W[k*n + i, r]

Note the W index ordering: W[k*n + i, r] (output factor follows the official
trace convention W[k*n+i, r]).

The trivial schoolbook rank is n*m*p = 2*2*15 = 60.

We can do much better.  For a 2x2 by 2x15 multiplication, the tensor is a sum
of 15 "slices" each corresponding to a fixed output column k.  Each slice is a
2x2 by 2x1 (outer product) tensor of rank at most 2.  So the total rank is at
most 15*2 = 30.

Construction for rank R = 2*p = 30:
For each k in range(p):
  For each j in range(m):  (m=2)
    r = k*m + j
    U[i*m + j, r] = 1 for all i
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1 for all i

Wait, let me verify: we need sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r].

For a fixed (i, j, k), the LHS sums over r.  Only r values where both U[i*m+j,r]
and V[j*p+k,r] are nonzero contribute.

With r = k'*m + j':
  U[i*m + j, r] = 1  iff  j' = j  (regardless of i and k')
  V[j*p + k, r] = 1  iff  j' = j AND k' = k

So the only r that contributes is r = k*m + j.  Then:
  LHS = U[i*m+j, k*m+j] * V[j*p+k, k*m+j] = 1 * 1 = 1

And W[k*n + i, r] should equal 1 when r = k*m+j and 0 otherwise.

So set W[k*n + i, r] = 1 iff r = k*m + j for some j (i.e., r mod m = j and r div m = k).
More precisely, W[k*n + i, r] = 1 if floor(r/m) == k and r % m == j... but j is free.

Actually W[k*n + i, r] = 1 iff floor(r/m) == k.  (The r%m part is handled by U and V.)

Let me re-examine: For W[k*n + i, r], we set it to 1 when floor(r/m) == k.  This means
for a given output column k, there are m=2 columns in W (r = k*2 and r = k*2+1) that are 1.

Check: sum_r U[i*m+j, r] * V[j*p+k, r]:
- U[i*m+j, r] = 1 iff r % m == j  (i.e., r % 2 == j)
- V[j*p+k, r] = 1 iff r // m == k AND r % m == j

Wait, I need to be more careful.  Let me redefine:

r = k'*m + j'  where k' in [0,p), j' in [0,m).

U[i*m + j, r] = 1 iff j' == j  (i.e., r % m == j)
V[j*p + k, r] = 1 iff k' == k AND j' == j  (i.e., r // m == k AND r % m == j)

So both are 1 iff r // m == k AND r % m == j, i.e., r == k*m + j.

LHS = 1 (exactly one term).

W[k*n + i, r] should be 1 when r == k*m + j for the specific j that appears in U and V.
But W doesn't depend on j in its index — it's W[k*n + i, r].

Hmm, the issue is that W[k*n+i, r] must equal 1 for the r that makes the product nonzero.
For a fixed (i, j, k), the nonzero r is r = k*m + j.  So W[k*n + i, k*m + j] = 1.

But for different j, we get different r.  So W[k*n + i, r] = 1 for r in {k*m, k*m+1} (both j values).

Let me verify the full identity:
For all i, j, k:
  sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]?

No wait, the identity is:
  sum_r U[i*m+j, r] * V[j*p+k, r] should equal... what exactly?

The tensor decomposition means:
  T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

No, that's not right either.  In a standard tensor decomposition of matrix multiplication:

C[i,k] = sum_j A[i,j] B[j,k]

The tensor T has T[i,j,k] = delta (it's 1 for the "multiplication" pattern).  Actually the
matmul tensor is defined as T[i,j,k] = 1 (it's the structural tensor).  The decomposition is:

T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait no.  The standard formulation: we want to express the matmul as a sum of rank-1 tensors.
A rank-1 term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].  The sum over r gives the tensor.

So the condition is:
  For all (i, j, k):  T[i,j,k] = sum_r  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

where T[i,j,k] = 1 for all i,j,k (the matmul tensor is the all-ones tensor of shape (n*m, m*p, p*n)... no).

Actually, I think the task defines the decomposition differently.  Let me re-read:

"U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries
that exactly decompose the matrix multiplication tensor"

The standard matmul tensor decomposition:
  sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]

where a = i*m+j, b = j*p+k, c = k*n+i, and T[a,b,c] = 1 iff the indices are consistent
(i.e., the j in a matches the j in b, and the k in b matches the k in c, and the i in c
matches the i in a).

Actually for the matmul tensor, T[i,j,k] = 1 for all i,j,k.  The "tensor" here is just
the identity mapping.  Let me think again.

The matrix multiplication can be written as:
  C = A * B

In tensor form, the multiplication is a bilinear map (A, B) -> C.  The tensor T is a
3-tensor of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1.  Wait, that's not
a standard 3-tensor because the indices are coupled.

Actually, I think the correct interpretation is:

The matmul tensor T is of shape (n*m, m*p, p*n).  T[a, b, c] = 1 iff:
  a = i*m + j, b = j'*p + k, c = k'*n + i'
  with i=i', j=j', k=k'

No, that makes it a permutation-like tensor.  Let me just use the standard formulation.

Standard: The bilinear map (A,B) -> C = A*B can be written as:
  C[i,k] = sum_{j} sum_{r}  U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]

Wait, I think the convention is:
  C = sum_r  (U[:,r] outer A) * (V[:,r] outer B) ... no.

Let me just look at the schoolbook:
  For each (i,j,k):  U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.

The product U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for the matching (i,j,k) and 0 otherwise.
So the sum over r gives: for each (i,j,k), the tensor entry is 1.

So the condition is:
  For all (i,j,k):  sum_r  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  ==  1

And for "non-matching" triples (where the indices don't form a valid (i,j,k) chain), the sum should be 0.  But since i,j,k range over all valid values, every (a,b,c) = (i*m+j, j*p+k, k*n+i) is a valid triple.

Hmm, but (a,b,c) is not uniquely determined by (i,j,k) because different (i,j,k) can give the same (a,b,c)?  No: a = i*m+j determines i and j.  b = j*p+k determines j and k.  c = k*n+i determines k and i.  So (a,b,c) is valid iff the j from a equals the j from b, and the k from b equals the k from c, and the i from c equals the i from a.

So the tensor T[a,b,c] = 1 iff the indices are consistent, 0 otherwise.  And we need:
  sum_r  U[a,r] * V[b,r] * W[c,r]  ==  T[a,b,c]  for all a,b,c.

For the schoolbook, this works because each (i,j,k) gets its own r.

Now for the rank-30 construction:
  r = k*m + j  (0 <= r < p*m = 30)

  U[i*m+j, r] = 1  iff  (r % m) == j
  V[j*p+k, r] = 1  iff  (r // m) == k  AND  (r % m) == j
  W[k*n+i, r] = 1  iff  (r // m) == k

Let's verify: For consistent (i,j,k), the product U*V*W:
  U[i*m+j, r] = 1 iff r%m == j
  V[j*p+k, r] = 1 iff r//m == k AND r%m == j
  W[k*n+i, r] = 1 iff r//m == k

Product is 1 iff r%m == j AND r//m == k, i.e., r == k*m + j.  Exactly one such r exists.
Sum = 1. ✓

For inconsistent (a,b,c):
  Let a = i*m+j, b = j2*p+k, c = k2*n+i2.
  Inconsistent means j != j2 or k != k2 or i != i2.

  We need sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r] == 0.

  U[i*m+j, r] = 1 iff r%m == j
  V[j2*p+k, r] = 1 iff r//m == k AND r%m == j2
  W[k2*n+i2, r] = 1 iff r//m == k2

  Product is 1 iff r%m == j AND r//m == k AND r%m == j2 AND r//m == k2.
  This requires j == j2 and k == k2.  If j != j2 or k != k2, product is always 0.

  If j == j2 but k != k2: need r%m == j AND r//m == k AND r//m == k2.  Since k != k2, impossible.
  If k == k2 but j != j2: need r%m == j AND r%m == j2.  Since j != j2, impossible.

So the rank-30 construction is valid!

Let me code it up.
"""

DIMENSIONS = [2, 2, 15]


def construct():
    n, m, p = DIMENSIONS
    rank = p * m  # 15 * 2 = 30

    # U: (n*m) x rank = 4 x 30
    # V: (m*p) x rank = 30 x 30
    # W: (p*n) x rank = 30 x 30

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        k_r = r // m   # which output column
        j_r = r % m    # which inner index

        # U[i*m + j, r] = 1  iff j == j_r
        for i in range(n):
            for j in range(m):
                if j == j_r:
                    U[i * m + j][r] = 1

        # V[j*p + k, r] = 1  iff j == j_r AND k == k_r
        for j in range(m):
            for k in range(p):
                if j == j_r and k == k_r:
                    V[j * p + k][r] = 1

        # W[k*n + i, r] = 1  iff k == k_r
        for k in range(p):
            for i in range(n):
                if k == k_r:
                    W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
