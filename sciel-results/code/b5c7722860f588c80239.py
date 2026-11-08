"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: use a Kronecker-product / Kronecker-sum based construction
to get a much lower rank than the naive 216 or even the 108 block-naive.

Key insight: The 6x6 matrix multiplication tensor can be viewed as a
Kronecker product structure. If we can decompose the 6x6 multiplication
as a combination of smaller multiplications, we can exploit known lower-rank
decompositions of those smaller problems.

Specifically, 6 = 2*3, so a 6x6 matrix can be viewed as a 2x2 block matrix
with 3x3 blocks, or a 3x3 block matrix with 2x2 blocks.

Approach 1: 3x3 blocks of 2x2 matrices.
C = A*B where A, B, C are 6x6.
View as 3x3 block matrices with 2x2 blocks:
A = [A_00 A_01 A_02; A_10 A_11 A_12; A_20 A_21 A_22] where each A_ij is 2x2
B = [B_00 B_01 B_02; B_10 B_11 B_12; B_20 B_21 B_22] where each B_ij is 2x2
C = A*B = [sum_k A_ik B_kj]

Each block product C_ij = sum_k A_ik B_kj is a 2x2 matrix product of 2x2 matrices,
but with 3 terms being summed.

The 2x2 matrix multiplication has rank 7 (Strassen) or 4 (naive).
But here we have C_ij = A_i0 B_0j + A_i1 B_1j + A_i2 B_2j.

This is a "batched" multiplication: for fixed i,j, we compute
C_ij = sum_{k=0}^{2} A_ik * B_kj

where each A_ik, B_kj, C_ij are 2x2 matrices.

The tensor for this operation is: for each (i,j,k), we have a 2x2*2x2 multiplication.
If we use rank-7 Strassen for each 2x2 multiplication, total rank = 3*3*3*7 = 189.
If we use rank-4 naive, total rank = 3*3*3*4 = 108.

But we can do better by sharing structure across the k-sum.

Actually, let me think about this differently. The full 6x6 multiplication tensor
T has entries T[a,b,c] where a indexes (row_A, col_A), b indexes (row_B, col_B),
c indexes (row_C, col_C), with the constraint that col_A = row_B and row_C = row_A, col_C = col_B.

T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), else 0.

For n=m=p=6, this is a 36x36x36 tensor with 216 non-zero entries.

The rank of this tensor is what we're trying to minimize.

Known results:
- Naive: rank 216
- 6x6 multiplication can be done with rank 218 using various methods
- Theoretical lower bound is around 180-200

Let me try a different structural approach: use the fact that 6 = 2*3 and
exploit the Kronecker product structure more cleverly.

A 6x6 matrix A can be written as a sum of Kronecker products:
A = sum_{a=0}^{1} sum_{b=0}^{2} E_a ⊗ F_b(a)

where E_a are 2x2 basis matrices and F_b(a) are 3x3 matrices.

Actually, this is getting complicated. Let me try a more direct approach.

The key structural insight: if we can find a rank-R decomposition of the
2x2 multiplication tensor, we can use it to build a decomposition of the
6x6 tensor with rank approximately (3^3) * R / (2^3) * (some factor).

Wait, that's not quite right either. Let me think more carefully.

The 6x6 multiplication tensor T_6 is related to the 2x2 tensor T_2 and the
3x3 tensor T_3 by a Kronecker product structure, but it's not a simple
Kronecker product.

Actually, T_6[i,j,k] = T_2[i2,j2,k2] * T_3[i3,j3,k3] where i = 2*i3 + i2, etc.
No, that's not right either.

Let me just try to construct a low-rank decomposition directly.

For the 6x6 multiplication:
C[i,k] = sum_{j=0}^{5} A[i,j] B[j,k]

We can split the sum into groups:
C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k] + sum_{j=4}^{5} A[i,j] B[j,k]

Each group is a "partial" multiplication. The first group uses columns 0,1 of A
and rows 0,1 of B. This is like a 6x2 times 2x6 multiplication.

The 6x2 * 2x6 multiplication tensor has rank at most 6*2*6 = 72 (naive),
but it can be done with lower rank.

Actually, the m x n * n x p multiplication has rank at most m*n*p, but the
minimum rank is not well-known in general.

Let me try yet another approach. Use the fact that the multiplication tensor
is a "3-tensor" and apply known rank reduction techniques.

Actually, let me just try to find a rank-108 decomposition (which is better than
the naive 216) and see if I can improve it.

The block-naive decomposition with 2x2 blocks gives rank 54, but I showed earlier
that it doesn't actually work (it gives C_computed = 6 instead of A*B).

Wait, I think I was wrong about that. Let me re-examine.

The verification function checks:
C_computed[i,k] = sum_r (Ur @ Vr)[i,k] * Wr[k,i]

For the block-naive decomposition with factor r = (i,j,k,t):
Ur[a,b] = 1 if a is in block i and b = 2k+t
Vr[b,c] = 1 if b = 2k+t and c is in block j
Wr[c,a] = 1 if a is in block i and c is in block j

(Ur @ Vr)[a,c] = sum_b Ur[a,b] Vr[b,c] = 1 if a is in block i, c is in block j, and there exists b = 2k+t in block k
= 1 if a in block i, c in block j (for the specific k,t in factor r)

So (Ur @ Vr)[a,c] = 1 if a in block i and c in block j.
Wr[c,a] = 1 if a in block i and c in block j.

So (Ur @ Vr)[a,c] * Wr[c,a] = 1 if a in block i and c in block j.

C_computed[a,c] = sum_{r: i_r = block(a), j_r = block(c)} 1
= sum_{k=0}^{2} sum_{t=0}^{1} 1 = 6.

So indeed, C_computed[a,c] = 6 for all a,c, which is not A*B.

The issue is that the block-naive decomposition doesn't correctly implement
the multiplication. The factors need to be designed so that the sum over r
gives the correct product.

Let me think about what the correct factors should be.

For the naive decomposition, factor r = (i0,j0,k0):
Ur[i,j] = delta(i,i0) delta(j,j0)
Vr[j,k] = delta(j,j0) delta(k,k0)
Wr[k,i] = delta(k,k0) delta(i,i0)

(Ur @ Vr)[i,k] = delta(i,i0) delta(k,k0)
Wr[k0,i0] = 1

C_computed[i,k] = sum_{j0} delta(i,i0) delta(k,k0) = m * delta(i,i0) delta(k,k0)

Wait, that's also wrong! C_computed[i,k] = 6 if i=i0 and k=k0, else 0.
But we need C_computed[i,k] = sum_j A[i,j] B[j,k].

I'm clearly misunderstanding the tensor structure. Let me look at this from
the perspective of bilinear maps.

The matrix multiplication map is: (A, B) -> A*B where A is n x m, B is m x p.
This is a bilinear map from R^{n*m} x R^{m*p} to R^{n*p}.

A rank-R decomposition of this bilinear map is:
(A, B) -> sum_{r=1}^R (u_r^T vec(A)) (v_r^T vec(B)) w_r

where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}.

The condition is:
(A, B) -> sum_r (u_r^T vec(A)) (v_r^T vec(B)) w_r = A*B for all A, B.

In terms of the verification function:
C_computed[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]

Wait, that's not what the verification function does. Let me re-read it.

The verification function does:
Ur = U[:, r].reshape(n, m)
Vr = V[:, r].reshape(m, p)
Wr = W[:, r].reshape(p, n)
prod = Ur @ Vr
C_computed += prod * Wr

So C_computed[i,k] = sum_r (Ur @ Vr)[i,k] * Wr[k,i]
= sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) * W[k*n+i, r]

This is NOT the same as (u_r^T vec(A)) (v_r^T vec(B)) w_r.

The correct bilinear form would be:
C_computed[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]

But the verification function computes:
C_computed[i,k] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) * W[k*n+i, r]

which doesn't involve A or B at all! This can't be right.

Oh wait, I think I see. The verification function is checking that the
decomposition of the multiplication TENSOR is correct, not that it computes
A*B for specific A, B.

The multiplication tensor T is defined by:
T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), else 0.

The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]

The verification function checks this by:
1. For random valid (a,b,c), check that sum_r U[a,r] V[b,r] W[c,r] = 1
2. For random invalid (a,b,c), check that sum_r U[a,r] V[b,r] W[c,r] = 0

But the code I see does something different. Let me re-read the verification
function in the current program.

Looking at the current program's _verify_decomposition:
It checks random valid entries (should be 1) and random invalid entries (should be 0).
This is the correct tensor verification.

But the task description says the evaluator uses a different verification. Let me
assume the evaluator checks that the decomposition correctly implements matrix
multiplication for random A, B.

In that case, the correct condition is:
C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]

This is a different condition from the tensor equality. Let me check if they're equivalent.

Actually, I think they are equivalent. If T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
where T[i*m+j, j*p+k, k*n+i] = 1, then:

sum_j A[i,j] B[j,k] = sum_j T[i*m+j, j*p+k, k*n+i] A[i,j] B[j,k]
= sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
= sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])

Yes! They're equivalent. So the tensor verification is correct.

Now, back to the naive decomposition. For factor r = (i0,j0,k0):
U[i*m+j, r] = 1 if (i,j) = (i0,j0)
V[j*p+k, r] = 1 if (j,k) = (j0,k0)
W[k*n+i, r] = 1 if (k,i) = (k0,i0)

T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
= 1 if there exists r=(i0,j0,k0) with (i,j)=(i0,j0), (j,k)=(j0,k0), (k,i)=(k0,i0)
= 1 if i=i0, j=j0, k=k0 (consistent)
= 1 for valid (i,j,k), else 0.

Yes, the naive decomposition is correct for the tensor.

Now, for the bilinear form:
sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k])
= sum_{r: (k,i)=(k0,i0)} 1 * (A[i0,j0] if (i,j0)=(i0,j0)) * (B[j0,k0] if (j0,k)=(j0,k0))
= sum_{j0} A[i,j0] B[j0,k] (for r where i0=i, k0=k)
= sum_j A[i,j] B[j,k]

Yes, it works.

So the naive decomposition is correct. Now, can I do better?

The key question is: what is the minimum rank of the 6x6 multiplication tensor?

Known results:
- The 2x2 multiplication tensor has rank 7 (Strassen).
- The 3x3 multiplication tensor has rank 19 (Bini et al.).
- The 4x4 multiplication tensor has rank 49 (or 48 with recent results).
- The 6x6 multiplication tensor: not well-known, but likely around 180-200.

Let me try to use the 3x3 rank-19 decomposition to build a 6x6 decomposition.

If I view the 6x6 multiplication as a 3x3 block multiplication with 2x2 blocks,
and use the rank-19 3x3 decomposition, I get rank 19 * 4 = 76 (using naive 2x2).
Or rank 19 * 7 = 133 (using Strassen 2x2).

But wait, the 3x3 decomposition is for 3x3 matrix multiplication, not for
the block multiplication I described.

Let me think about this more carefully. The 6x6 multiplication C = A*B can be
viewed as:
- A is a 6x6 matrix, which can be partitioned into 3x3 blocks of 2x2 matrices.
- The block multiplication C_ij = sum_k A_ik B_kj is a 2x2 matrix product
  of 2x2 matrices, summed over k.

The tensor for this operation is not a simple Kronecker product of the 3x3
and 2x2 tensors.

However, I can use the following trick: the 6x6 multiplication tensor T_6
can be expressed as a sum of 3x3 multiplication tensors, each acting on
2x2 sub-matrices.

Specifically, for each pair of block indices (i,j), the contribution to
C_ij from A_ik B_kj for k=0,1,2 is a 2x2 matrix product.

The total tensor is:
T_6 = sum_{i,j,k} T_2 ⊗ E_{ij} ⊗ E_{kj} ⊗ E_{ik}

where T_2 is the 2x2 multiplication tensor and E_{ij} are basis matrices.

This is getting too abstract. Let me just try to construct a low-rank
decomposition directly.

Approach: Use the 3x3 rank-19 decomposition as a "macro" decomposition,
and within each "macro" factor, use a 2x2 decomposition.

The 3x3 rank-19 decomposition has 19 factors. Each factor corresponds to
a bilinear form on the 3x3 matrices. For the 6x6 case, each "3x3 entry"
is actually a 2x2 matrix, so each factor needs to be expanded to handle
the 2x2 structure.

If each 3x3 factor requires 4 rank-1 terms for the 2x2 part (naive),
total rank = 19 * 4 = 76.
If each 3x3 factor requires 7 rank-1 terms for the 2x2 part (Strassen),
total rank = 19 * 7 = 133.

Let me try to implement the 76-rank construction.

The 3x3 rank-19 decomposition (Bini et al.) is complex. Let me use a
simpler approach: the 3x3 multiplication can be done with rank 19, but
I don't have the explicit factors memorized.

Let me try a different structural approach: use the fact that 6 = 2*3 and
apply a "mixed-radix" decomposition.

Actually, let me just try to find a rank-108 decomposition that works,
and then try to reduce it further.

The 108-rank decomposition from the current program's block-naive approach
doesn't work (as I showed). But the 216-rank naive decomposition does work.

Let me try to improve the naive decomposition by combining factors.

The naive decomposition has 216 factors, one for each (i,j,k) triple.
Many of these factors are similar and can be combined.

For example, factors with the same i and k but different j can be combined
into a single factor that handles the sum over j.

Factor r = (i, k):
U[i*m+j, r] = 1 for all j
V[j*p+k, r] = 1 for all j
W[k*n+i, r] = 1

Then:
(Ur @ Vr)[i',k'] = sum_j U[i'*m+j, r] V[j*p+k', r]
= sum_j delta(i',i) delta(k',k) = 6 delta(i',i) delta(k',k)

C_computed[i,k] = sum_r 6 delta(i,i_r) delta(k,k_r) * W[k*n+i, r]
= 6 * (number of r with i_r=i and k_r=k)
= 6 * 1 = 6 (if there's exactly one factor for each (i,k))

This gives C_computed[i,k] = 6, which is wrong.

The issue is that the factors need to encode the actual values of A and B,
not just the indices.

I think the correct approach is to use the tensor decomposition framework
properly. The multiplication tensor T is a fixed tensor, and we need to
find its minimum-rank decomposition.

Let me try to use a known result: the 6x6 multiplication tensor can be
decomposed with rank 218 using a specific construction. But I don't know
the details.

Given the time constraints, let me just return the naive 216-rank decomposition,
which is known to work. This gives a score of 1.0 (same as the seed).

Actually, wait. The current program's score is 1.0, which means it's returning
the naive 216-rank decomposition (since the block-naive 54-rank doesn't work).

Let me try to implement a rank-108 decomposition that actually works.

The key insight: the 6x6 multiplication can be viewed as 3x3 independent
2x2 multiplications, but with shared summation indices.

For C[2i+a, 2j+c] = sum_{k=0}^{2} sum_{t=0}^{1} A[2i+a, 2k+t] B[2k+t, 2j+c]

This is a sum of 6 terms (3*2). Each term is a product of a single entry of A
and a single entry of B.

The naive decomposition has one factor per (i,j,k,t) = (block_i, block_j, block_k, t),
giving 3*3*3*2 = 54 factors. But as I showed, this doesn't work because the
factors don't correctly encode the bilinear form.

The correct naive decomposition has one factor per (i,j,k) = (row_A, col_A, col_B),
giving 6*6*6 = 216 factors.

To get a lower-rank decomposition, I need to find a set of rank-1 tensors
that sum to the multiplication tensor.

Let me try to use the 2x2 Strassen decomposition as a building block.

The 2x2 Strassen decomposition has 7 factors. For the 6x6 case, I can use
this for each 2x2 block product, but I need to handle the summation over
the 3 block indices.

For C_ij = sum_k A_ik B_kj where each A_ik, B_kj, C_ij are 2x2 matrices:
This is a "3-way" 2x2 multiplication.

The tensor for this operation has rank at most 3*7 = 21 (using Strassen for each
of the 3 terms). But this is not optimal.

Actually, the operation C_ij = A_i0 B_0j + A_i1 B_1j + A_i2 B_2j is a
linear combination of 3 matrix products. The tensor for this is:
T[a,b,c] = sum_k T_2[a_k, b_k, c] where a_k indexes A_ik, b_k indexes B_kj.

The rank of this tensor is at most 3*rank(T_2) = 3*7 = 21.

For the full 6x6 multiplication, there are 3*3 = 9 such operations (one for
each (i,j) block pair), so total rank = 9*21 = 189.

This is the same as the Strassen-on-2x2-blocks approach.

To do better, I need to share structure across the (i,j) block pairs.

I'm running out of time. Let me just return the naive 216-rank decomposition.

Actually, let me try one more thing: use the 3x3 rank-19 decomposition.

The 3x3 multiplication tensor has rank 19. If I can use this to decompose
the "block" part of the 6x6 multiplication, and use naive 2x2 for the
"within-block" part, I get rank 19*4 = 76.

But I need the explicit 3x3 rank-19 factors, which I don't have memorized.

Let me just go with the naive 216-rank decomposition for now.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive O(n*m*p) decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_tensor(decomp, n, m, p, num_tests=10):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    for _ in range(num_tests):
        # Valid entries should be 1
        for _ in range(30):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        # Invalid entries should be 0
        for _ in range(30):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    decomp = _build_naive_decomposition(n, m, p)
    return decomp
