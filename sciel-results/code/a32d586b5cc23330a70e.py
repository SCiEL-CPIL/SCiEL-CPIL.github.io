"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Uses Strassen's 7-term decomposition for 2x2 sub-blocks applied to the first
two columns of A (paired with the first two rows of B), and schoolbook for the
remaining column (column 2 of A with all 3 rows of B).

Decomposition:
  M_{2,3,8} -> (M_{2,2,4} + M_{2,2,4}) + (M_{2,1,8})
              -> 28 + 28 + 16 = 72

Actually, for the first 2 rows of B (k=0..3), we split into two 2x2x4 blocks:
  Block 1: A[:,0:2] * B[0:2, 0:2]  -> 2x2 * 2x2 -> rank 7
  Block 2: A[:,0:2] * B[0:2, 2:4]  -> 2x2 * 2x2 -> rank 7
  Block 3: A[:,0:2] * B[0:2, 4:6]  -> 2x2 * 2x2 -> rank 7
  Block 4: A[:,0:2] * B[0:2, 6:8]  -> 2x2 * 2x2 -> rank 7
  Remaining: A[:,2] * B[2,:]        -> 2x1 * 1x8 -> rank 16
  Total: 4*7 + 16 = 44

Wait, that's still 44. Let me reconsider the structure.

The matrix multiplication M_{2,3,8}:
  C = A * B where A is 2x3, B is 3x8, C is 2x8
  C[i,k] = sum_{a=0}^{2} A[i,a] * B[a,k]

The tensor has entries T[i*3+a, a*8+k, k*2+i] = 1 for all i,a,k.

One approach: view the last row (a=2) of A separately.
- The part involving a=0,1 is a 2x2 matrix (A[0:2,0:2]) times B[0:2,:], giving 2x8.
  This is M_{2,2,8} with rank 28 (4 blocks of Strassen 2x2x2).
- The part involving a=2 is 2x1 (A[:,2]) times 1x8 (B[2,:]), giving 2x8.
  This is M_{2,1,8} with rank 16 (each of 8 outputs needs 2 terms).
- Total: 28 + 16 = 44.

Can we do better? Let's try to share terms between the two parts.

Alternative: treat it as a 2x3x8 tensor. The known rank of M_{2,3,8}...

Actually, for M_{2,n,p}, the rank is at most 2*p + 4*(n-2) for large p, but that's not tight.

Let me think about the slice rank. Fixing k, we have a bilinear map:
  C_k = sum_a A[:,a] * B[a,k] = A * B[:,k]
This is an outer product of a 2-vector and a 3-vector, so rank 3 per slice.
Total slice rank upper bound: 8 * 3 = 24? No, that's not how it works.

Actually, the "slice rank" bound: for M_{m,n,p}, the rank is at least max(mn, np, pm).
For 2,3,8: max(6, 24, 16) = 24. So rank >= 24.

But can we achieve rank 24? For M_{2,3,8}, one approach:
- View as C[i,k] = sum_{a=0}^{2} A[i,a] * B[a,k]
- For each k, this is a bilinear form in A and B[:,k].

Let me try a different decomposition. 

For M_{2,3,8}, we can use the following:
The 8 output columns each depend on all 6 entries of A and the 3 entries of the corresponding column of B.

Actually, let me reconsider. The standard decomposition gives rank 44 as above.

Can we overlap the Strassen decomposition across the 4 blocks? 

Consider the four 2x2x2 problems:
  For block j (j=0..3), we compute A[0:2,0:2] * B[0:2, 2j:2j+2]
  
The A factor is the same for all 4 blocks! So the U factor for the 4 Strassen decompositions 
can potentially be shared.

For Strassen 2x2x2, the U factor has 7 rows (one per term), with values in {0,1,-1}.
The V factor depends on B[0:2, 2j:2j+2], which is different for each j.
The W factor depends on the output, which is also different.

If we use the same 7 Strassen terms for all 4 blocks, we'd have 28 terms total.
But the U factor is the same across all 4 blocks. However, in a tensor decomposition,
each term is a rank-1 tensor, so we can't "share" the U part across different terms.

Unless we group them: for each Strassen term r (0..6), we have 4 instances (one per block j).
Each instance has the same U vector but different V and W vectors.

So we can't reduce below 28 for the A[0:2,0:2]*B[0:2,:] part.

Total remains 28 + 16 = 44.

Hmm, but maybe there's a smarter way to handle the boundary between the Strassen part and the schoolbook part.

Actually, let me just go with rank 44, which is what the current program achieves.

The key insight from the directive: "Model the multiplication tensor as a 3-dimensional array and apply the slice rank bound by fixing the output index k."

For each k (0..7), the output C[:,k] = A * B[:,k], which is a 2-vector.
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]

This is a bilinear map from (A[0,*], A[1,*]) x B[:,k] to C[:,k].
The rank of this 2x3 -> 2 map is... it's equivalent to computing two dot products.

For each k, we need 6 terms in general. But we can share across k.

Let me try a construction with rank 24:
For each a (0,1,2) and each k (0..7), we could have one term, but that's 24.
Term (a,k): U[i*m+a] = delta_{i,*}... no, U is 6x24, V is 24x24, W is 16x24.

Actually, U has shape (n*m, rank) = (6, 24)
V has shape (m*p, rank) = (24, 24)
W has shape (p*n, rank) = (16, 24)

For term (a,k) with a in 0..2 and k in 0..7:
  U[i*m + a] = 1 for all i? No, U[i*m+a] should be 1 for the specific i.

Wait, let me re-read the interface.

U has shape (n*m, rank) = (6, rank)
V has shape (m*p, rank) = (24, rank)
W has shape (p*n, rank) = (16, rank)

The decomposition: sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c] = 1 for all i,a,k.

So for each (i,a,k), the sum over c of U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] should equal 1.

For schoolbook with rank 48:
  For each (i,a,k), create a term c where U[i*m+a,c]=1, V[a*p+k,c]=1, W[k*n+i,c]=1.

For rank 24, we'd need to group these.

Let me try: for each a and k, create one term. Then U[i*m+a,c_{a,k}] = delta_{i,?}
But U depends on i, so we need U[i*m+a, c] to depend on i.

Actually, for fixed a,k, the terms need to satisfy:
  sum_c U[i*m+a,c] * V[a*p+k,c] * W[k*n+i,c] = 1 for i=0,1

If we use one term per (a,k), then U[i*m+a, c_{a,k}] * V[a*p+k, c_{a,k}] * W[k*n+i, c_{a,k}] = 1 for i=0,1.
This requires U[0*m+a, c] = U[1*m+a, c] and V[a*p+k, c] * W[k*n+0, c] * U[a, c] = 1 and V[a*p+k, c] * W[k*n+1, c] * U[m+a, c] = 1.
So U[a,c] * V[a*p+k,c] * W[k*n+0,c] = 1 and U[m+a,c] * V[a*p+k,c] * W[k*n+1,c] = 1.
This is possible with U[a,c]=1, U[m+a,c]=1, V[a*p+k,c]=1, W[k*n+0,c]=1, W[k*n+1,c]=1.
But then the term contributes to both i=0 and i=1 simultaneously.

For each (a,k), one term with U[i*m+a,c]=1 for both i, V[a*p+k,c]=1, W[k*n+i,c]=1 for both i.
But W[k*n+i, c] would be 1 for both i, so the term contributes 1 to C[0,k] and 1 to C[1,k] when multiplied by U and V.

Wait, the decomposition is:
  C[i,k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]

No, that's not right. Let me re-read.

"The output factor follows the official trace convention W[k*n+i,r]."

So W has indices (k*n+i) where k is 0..p-1 and i is 0..n-1.

The decomposition for matrix multiplication:
  (A*B)[i,k] = sum_{a} A[i,a] * B[a,k]

In tensor form, the multiplication tensor T has T[i,a,k] = delta (it's 1 for all valid i,a,k).

The rank-r decomposition means:
  T[i,a,k] = sum_{r=0}^{rank-1} U[i,a,r] * V[a,k,r] * W[k,i,r]

Wait, but the shapes are:
  U: (n*m, rank) -> U[i*m+a, r]
  V: (m*p, rank) -> V[a*p+k, r]
  W: (p*n, rank) -> W[k*n+i, r]

And the condition is:
  sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] = 1 for all i,a,k

Yes, that's correct.

So for schoolbook, each (i,a,k) gets its own term.

For a grouped approach: if we fix a and k, and let the term contribute to both i=0 and i=1:
  For i=0: U[0*m+a, r] * V[a*p+k, r] * W[k*n+0, r] 
  For i=1: U[1*m+a, r] * V[a*p+k, r] * W[k*n+1, r]

If we set U[0*m+a, r] = 1, U[1*m+a, r] = 1, V[a*p+k, r] = 1, W[k*n+0, r] = 1, W[k*n+1, r] = 1,
then both i=0 and i=1 get contribution 1 from this term.

But we need each (i,a,k) to get exactly 1 total. So if one term contributes 1 to both i=0 and i=1 for a given (a,k), that's fine as long as no other term contributes to (i,a,k).

So rank 24 would work with one term per (a,k) pair! Let me verify.

For term (a,k): 
  U[i*m+a, r] = 1 for i=0,1
  V[a*p+k, r] = 1
  W[k*n+i, r] = 1 for i=0,1

Then for any (i,a',k'):
  sum_r U[i*m+a', r] * V[a'*p+k', r] * W[k'*n+i, r]
  = 1 if (a',k') = (a,k) else 0

Wait, that's not right. The sum is over r, and for a fixed (i,a',k'), only the term r=(a',k') has U[i*m+a', r] = 1 (if a'=a and the term is for (a,k) with k=k').

Actually, for fixed (i,a',k'), the sum over r:
  Only r corresponding to (a',k') has V[a'*p+k', r] = 1 and U[i*m+a', r] = 1 and W[k'*n+i, r] = 1.
  All other r have at least one of these being 0.

So the sum is 1. 

This gives rank 24! Let me verify with a small example.

n=2, m=3, p=8
rank = 24

For r = a*p + k (wait, that's 24 values, and a is 0..2, k is 0..7, so a*8+k ranges 0..23).

U[i*m+a, r] where r = a*p+k:
  U[i*m+a, a*p+k] = 1 for i=0,1, and for the specific a.
  But U is 6x24. Row i*m+a.
  For row 0 (i=0,a=0): U[0, 0*8+k] = 1 for k=0..7? No, only U[0, 0*8+k] for the specific k.
  
  Actually, U[i*m+a, r] = 1 only when r = a*p+k and the row is i*m+a.
  For row i*m+a, the non-zero entries are at columns a*p+0, a*p+1, ..., a*p+7.
  All of these are 1!

So U[0, 0]=1, U[0,1]=1, ..., U[0,7]=1
   U[1, 0]=0, U[1,1]=0, ..., U[1,7]=0
   U[1, 8]=0, ..., etc.
   
Wait, U[i*m+a, r] where r = a'*p+k'. U[i*m+a, a'*p+k'] = 1 if a'=a, else 0.
So for row 0 (i=0, a=0): columns 0..7 are 1, columns 8..23 are 0.
For row 1 (i=0, a=1): columns 0..7 are 0, columns 8..15 are 1, columns 16..23 are 0.
For row 2 (i=0, a=2): columns 16..23 are 1.
For row 3 (i=1, a=0): columns 0..7 are 1.
For row 4 (i=1, a=1): columns 8..15 are 1.
For row 5 (i=1, a=2): columns 16..23 are 1.

V[a*p+k, r] where r = a'*p+k':
  V[a*p+k, a'*p+k'] = 1 if a'=a and k'=k, else 0.
  So V is the identity-like: V[a*p+k, a*p+k] = 1.

W[k*n+i, r] where r = a'*p+k':
  W[k*n+i, a'*p+k'] = 1 if k'=k, else 0.
  For row k*n+0: columns where k'=k, i.e., columns k*8, k*8+1, ..., k*8+7 are 1? No.
  
  Wait, r = a'*p+k' where p=8. So r = 8*a' + k'.
  W[k*n+i, r] = 1 if k'=k.
  For row k*n+i, the columns r where 8*a'+k' = r and k'=k.
  k' = k, so r = 8*a' + k for a'=0,1,2.
  So W[k*2+0, 8*a'+k] = 1 for a'=0,1,2, and 0 otherwise.
  Similarly W[k*2+1, 8*a'+k] = 1 for a'=0,1,2.

Now check: for (i=0, a=0, k=0):
  sum_r U[0, r] * V[0, r] * W[0, r]
  = sum_r U[0,r] * V[0,r] * W[0,r]
  U[0,r] is 1 for r=0..7.
  V[0,r] is 1 only for r=0.
  W[0,r] is 1 for r=0,8,16 (since k=0, k'=0, r=8*a'+0).
  So only r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. Product=1.
  Sum=1. Correct.

For (i=0, a=1, k=3):
  U[1,r] is 1 for r=8..15.
  V[3,r] is 1 only for r=3.
  But r=3 is not in 8..15. So sum=0. WRONG!

I made an error. V[a*p+k, r] should be 1 when r corresponds to (a,k).
V[1*8+3, r] = V[11, r] = 1 when r=11.
U[0*m+1, 11] = U[1, 11] = 1 (since a=1, r=8*1+3=11, and 11 is in 8..15).
W[3*2+0, 11] = W[6, 11] = 1 (since k=3, k'=3, r=8*a'+3, and 11=8*1+3, so a'=1, k'=3).

So for (i=0, a=1, k=3):
  r=11: U[1,11]=1, V[11,11]=1, W[6,11]=1. Product=1.
  Sum=1. Correct!

Let me re-check the general case:
  For (i,a,k):
  r = a*p + k = 8a + k
  U[i*m+a, r] = U[i*3+a, 8a+k] = 1 (since the row is i*3+a and r=8a+k, and 8a+k is in [8a, 8a+7])
  V[a*p+k, r] = V[8a+k, 8a+k] = 1
  W[k*n+i, r] = W[2k+i, 8a+k] = 1 (since k'=k, so r=8a'+k, and 8a+k matches with a'=a)

For any other r':
  If r' = 8a'+k' with (a',k') != (a,k):
  - If a' != a: U[i*3+a, 8a'+k'] = 0 (since 8a'+k' is not in [8a, 8a+7])
  - If a' == a but k' != k: V[8a+k, 8a+k'] = 0
  - If a' != a but k' == k: U[i*3+a, 8a'+k] = 0
  - If a' == a and k' == k: this is r' = r, already counted.

So the sum is exactly 1. Rank 24 works!

This is a valid rank-24 decomposition. The score would be seed/24.

Let me implement this.
"""

import time
import sys

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank-24 decomposition: one term per (a,k) pair
    # U[i*m+a, a*p+k] = 1
    # V[a*p+k, a*p+k] = 1
    # W[k*n+i, a*p+k] = 1
    rank = m * p  # 24

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for a in range(m):
        for k in range(p):
            r = a * p + k
            # U[i*m+a, r] = 1 for i in 0..n-1
            for i in range(n):
                U[i * m + a][r] = 1
            # V[a*p+k, r] = 1
            V[a * p + k][r] = 1
            # W[k*n+i, r] = 1 for i in 0..n-1
            for i in range(n):
                W[k * n + i][r] = 1

    if not _elapsed_ok():
        return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
