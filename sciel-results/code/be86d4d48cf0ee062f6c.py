"""Program for M_{2,3,9} matrix multiplication tensor decomposition.

Strategy: exploit the block structure of B (three 3x3 blocks) and use the
known rank-9 decomposition of M_{2,3,3} for each block, sharing the U-factors.

The key insight: C[i, 3t+r] = sum_j A[i,j] B[j, 3t+r]
For fixed t, this is a 2x3 by 3x3 multiplication. The U-factors (involving A)
are the same across all three blocks. So we can use a rank-9 decomposition for
each block and share the U part.

Actually, the cleanest approach: the full tensor M_{2,3,9} can be seen as
three copies of M_{2,3,3} acting on the same A but different B-blocks.
The rank is at most 3 * rank(M_{2,3,3}).

For M_{2,3,3}: the schoolbook gives rank 18. But we can do better.
Actually, for 2x3 * 3x3, the tensor has 2*3*3 = 18 entries in the output,
and the rank is known to be 9 (since it's a "rectangular" case where
min(n*p, m*p, n*m) considerations apply, but actually for 2x3x3 the rank
is 9).

Wait, let me reconsider. The standard schoolbook for 2x3 * 3x3:
C[i,k] = sum_j A[i,j] B[j,k] for i in 0..1, j in 0..2, k in 0..2
That's 2*3*3 = 18 terms. But we can combine them.

Actually, the rank of the matrix multiplication tensor M_{n,m,p} is at most
n*m*p (schoolbook). For M_{2,3,3}, schoolbook is 18. But the true rank is 9.

The rank-9 decomposition of M_{2,3,3}:
We can write C[i,k] = sum_j A[i,j] B[j,k].
This is equivalent to: for each (i,k), C[i,k] = <A_i, B^k> where A_i is row i of A
and B^k is column k of B.

The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * 1 = sum_{i,j,k} A_i_j B_j_k C_ik
As a bilinear form, T(A,B) = A * B.

The rank of this tensor: since A is 2x3 and B is 3x3, the product is 2x3.
The tensor lives in (R^2*3)^* ⊗ (R^3*3)^* ⊗ (R^3*2)^* ... 

Let me just use the fact that for 2x3 * 3x3, we can use rank 9:
Group the 18 schoolbook terms into 9 pairs. For each j (0,1,2) and each of
the 2*3 = 6 (i,k) pairs, we have a term. That's 3*6 = 18.

To get rank 9, we need to pair up terms. Notice that for fixed j, we have
6 terms (one for each (i,k)). We can't easily pair these.

Alternative: use the identity that M_{2,3,3} can be embedded in M_{3,3,3}
which has rank 21 (Strassen) or 18 (schoolbook). That doesn't help.

Let me just use rank 18 for each block, giving total rank 54. But that's the same as seed.

Wait, I need to think more carefully. For M_{2,3,9}:
C is 2x9, A is 2x3, B is 3x9.
C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]

The schoolbook rank is 2*3*9 = 54.

Can we do better? The lower bound is max(n*p, m*p, n*m) = max(18, 27, 6) = 27.
Actually, the lower bound for M_{n,m,p} is at least n*p (since the output has n*p entries
and each rank-1 term contributes at most 1 to the "output dimension" in some sense).

More precisely, the rank of M_{n,m,p} is at least n*p if m >= 1, and at least m*p if n >= 1,
etc. The best known lower bound is max(n*p, m*p, n*m, ...).

For M_{2,3,9}: lower bound is max(2*9, 3*9, 2*3) = max(18, 27, 6) = 27.

So rank is between 27 and 54.

Construction achieving rank 27:
For each of the 3 column-blocks t=0,1,2, we have a 2x3 * 3x3 multiplication.
If we can do each 2x3*3x3 in rank 9, and share the U-factors, we get rank 27.

But can we do 2x3*3x3 in rank 9? Let's see:
C[i,k] = sum_j A[i,j] B[j,k]
For i=0,1 and k=0,1,2, j=0,1,2.

The 18 schoolbook terms are: (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.

To get rank 9, we need to combine pairs. Consider:
For fixed j, the 6 terms (i,k) can be written as:
A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}

This is an outer product of the vector (A[0,j], A[1,j]) with (B[j,0], B[j,1], B[j,2]).
As a tensor, this contributes to the "slice" j of the tensor.

The full tensor is sum_j (a_j ⊗ b_j) where a_j = (A[0,j], A[1,j]) and b_j = (B[j,0], B[j,1], B[j,2]).
But this is in the space of 2x3 matrices, so each "term" j is a rank-1 matrix A[:,j] * B[j,:].
The sum of 3 rank-1 matrices. But we need the tensor decomposition, not matrix decomposition.

The tensor T = sum_{i,j,k} T_{ijk} e_i ⊗ e_j ⊗ e_k where T_{ijk} = 1 for all i,j,k.
Wait no, T_{ijk} corresponds to the coefficient of A[i,j] B[j,k] C[i,k].
So T = sum_{i,j,k} e_i^{(A)} ⊗ e_j^{(B)} ⊗ e_{(i,k)}^{(C)} where the index j is shared.

More carefully: T = sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{2} A_{ij} B_{jk} C_{ik}
= sum_{i,j,k} (e_i ⊗ e_j)_{A} ⊗ (e_j ⊗ e_k)_{B} ⊗ (e_i ⊗ e_k)_{C}

The rank of this tensor: it's the matrix multiplication tensor for 2x3 * 3x3.
Known result: rank(M_{2,3,3}) = 9.

The rank-9 decomposition:
C_{ik} = sum_{j} A_{ij} B_{jk}

We can write this as 9 terms by using the fact that for 2x3*3x3, we can use:
For each pair (i,k), C_{ik} = A_{i0}B_{0k} + A_{i1}B_{1k} + A_{i2}B_{2k}

That's 6*3 = 18 terms. To get 9, we need to combine.

Actually, I recall that the rank of M_{2,3,3} is indeed 9, and the decomposition is:
For t=0,1,2 and s=0,1,2, define:
U: A_{is} gets coefficient 1 for term (s,t)
V: B_{sk} gets coefficient 1 for term (s,t)  
W: C_{ik} gets coefficient 1 for term (s,t) where... 

Hmm, this doesn't quite work because the indices don't match.

Let me try a different approach. Since I can't easily find a rank-27 decomposition,
let me try to find a rank-30 or so decomposition.

Actually, let me just carefully implement the block structure with rank 18 per block
but sharing U factors. If U is shared, then for each block t, we use 18 terms but
the U part is the same. So total rank = 3 * 18 = 54 if we don't share, or less if we do.

Wait, if U is shared, it means the same U columns are used for all blocks. But the V and W
columns are different for each block. So the total rank is still 54 because each term
(i,j,k,t) is distinct.

I think the safest approach is to just output the schoolbook rank 54, but let me try
one more thing: can I use a rank-10 decomposition for 2x3*3x3?

Actually, I'll just go with a clean implementation of the schoolbook at rank 54,
since I can't find a valid lower-rank decomposition with certainty.

But wait - let me try rank 27 with a specific construction.

For M_{2,3,9}, write B as [B0 | B1 | B2] where each Bt is 3x3.
C = [A*B0 | A*B1 | A*B2]

For each t, A*Bt is a 2x3 matrix computed from A (2x3) and Bt (3x3).
The tensor for A*Bt has the same A part but different B and C parts.

If I use the schoolbook for each A*Bt (rank 18 each), total is 54.
If I can find a rank-9 decomposition for A*Bt, total is 27.

Let me try to construct rank 9 for 2x3*3x3:
C_{ik} = sum_j A_{ij} B_{jk}

Define 9 terms indexed by (j, s) where j in {0,1,2} and s in {0,1,2}:
U: for term (j,s), A_{is} gets... no, this doesn't work because i is free.

OK here's a valid rank-9 decomposition for 2x3*3x3:
For each j in {0,1,2} and each (i,k) pair... no that's 18.

I give up trying to find rank < 54 analytically. Let me just output rank 54 cleanly.
"""

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    R = n * m * p  # 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
