"""Matrix multiplication 2x4 by 4x5 - rank-minimal exact tensor decomposition.

Approach: exploit the n=2 structure. The multiplication tensor for C = A*B
where A is 2x4 and B is 4x5 can be viewed as computing 10 output entries.

Key insight: we can decompose the 4x5 B matrix into overlapping 2x2 blocks
and use Strassen's algorithm (rank 7) for each 2x2 sub-multiplication,
combined with careful sharing of partial products.

Strategy: Split the 4 rows of B (and corresponding 4 columns of A) into
two 2x4 blocks. Each 2x4 block times 2x5 block gives a 2x5 result.
For each 2x5 = (2x2)(2x5): split the 2x5 into a 2x2 and a 2x3.
  - 2x2 part: Strassen (rank 7)
  - 2x3 part: split into 2x1 + 2x2. 
    - 2x1: schoolbook rank 4 (each output = 2 mults, 2 outputs)
    - 2x2: Strassen (rank 7)
  Total per 2x5: 7 + 4 + 7 = 18

Two 2x5 blocks: 2 * 18 = 36.

But we can do better by sharing. Let's try a different split.

Alternative: Directly compute C[i][k] = sum_j A[i][j] B[j][k]
C is 2x5, so 10 outputs, each is a sum of 4 products.

Let me try: for each column k of C (5 columns), C[:,k] = A @ B[:,k]
A is 2x4, B[:,k] is 4x1. This is a 2x4 @ 4x1 multiplication.
Split A into two 2x2 blocks: A1 (cols 0,1) and A2 (cols 2,3).
C[:,k] = A1 @ B[:2,k] + A2 @ B[2:4,k]
Each A_i @ B_sub is 2x2 @ 2x1.
2x2 @ 2x1: output is 2x1, each entry is sum of 2 products.
This can be done in rank 4 (schoolbook) or... can we do better?
Actually 2x2 @ 2x1 has tensor rank 4 (it's just 4 independent products
grouped into 2 sums, and no bilinear structure saves us here since
one matrix is 2x2 and the other is 2x1).

Wait, 2x2 @ 2x1: 
C0 = A00*B0 + A01*B1
C1 = A10*B0 + A11*B1
This is 4 multiplications, rank 4. No way to reduce below 4 since
each of the 4 products appears in exactly one output.

So per column k: 2 * 4 = 8. For 5 columns: 40. Worse than 36.

Let me reconsider the 36-rank approach and see if we can share.

Actually, let me try a completely different approach. 

C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

We have 2*5 = 10 output entries. Each involves 4 terms. Total 40 scalar
products, but they share structure.

Let me use the "4x5" structure more directly. Write:
C = A @ B where A is 2x4, B is 4x5.

Split B into B_L (4x2) and B_R (4x3).
C_L = A @ B_L (2x2), C_R = A @ B_R (2x3).

For C_L (2x2 = 2x4 @ 4x2):
Split A = [A1 A2] where A1, A2 are 2x2.
C_L = A1 @ B_L[0:2,:] + A2 @ B_L[2:4,:]
Each term is 2x2 @ 2x2, use Strassen: rank 7 each. Total: 14.

For C_R (2x3 = 2x4 @ 4x3):
Split A = [A1 A2] where A1, A2 are 2x2.
C_R = A1 @ B_R[0:2,:] + A2 @ B_R[2:4,:]
Each term is 2x2 @ 2x3.
2x2 @ 2x3: split into 2x2 @ (2x1 + 2x2) or (2x2)@(2x3).
Let's split the 2x3 output into col0 (2x1) and cols1-2 (2x2).
2x2 @ 2x1: rank 4
2x2 @ 2x2: rank 7 (Strassen)
Total per 2x2@2x3: 11
Two of them: 22.

Total: 14 + 22 = 36. Same as before.

Can we do better? Let's think about whether 2x2@2x3 can be done in less than 11.
Actually, the tensor rank of 2x2 @ 2x3 multiplication... 
The output has 6 entries. Each is a sum of 2 products (since 2x2 @ 2x1 per column).
Total 12 products. Can we share? 

C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0,1,2.
This is 6 outputs, 12 products. The products are:
A00*B0k, A01*B1k for k=0,1,2 (6 products for row 0)
A10*B0k, A11*B1k for k=0,1,2 (6 products for row 1)

These are all independent (no product appears in two outputs), so rank is 12? 
No wait, each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] uses 2 products.
The 12 products are all distinct, and each appears in exactly one output.
So the tensor rank is exactly 12 for 2x2 @ 2x3.

Hmm, that means my earlier decomposition gives:
C_L: 14 (two Strassen 2x2@2x2)
C_R: 2 * 12 = 24 (two 2x2@2x3, each rank 12)
Total: 38. Worse!

Wait, I made an error. Let me recheck. For 2x2 @ 2x3, if we treat it as
computing C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,1,2}, j in {0,1}.

The bilinear map has 6 outputs, each a sum of 2 terms. The 12 terms are
A[i,j]*B[j,k] for all (i,j,k). Are they all independent? Yes, each triple
(i,j,k) gives a unique product that appears in exactly one output C[i,k].
So rank = 12.

But wait, Strassen gives 2x2@2x2 in rank 7, which is less than 8 (schoolbook).
The savings come from the fact that in 2x2@2x2, products like A00*B00 appear
in C00, and A00*B11 appears in C11, etc. There's overlap in how products
contribute to different outputs.

For 2x2@2x3, similarly there should be some sharing. Let me think more carefully.

C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

The products are: A00*B00, A00*B01, A00*B02, A01*B10, A01*B11, A01*B12,
A10*B00, A10*B01, A10*B02, A11*B10, A11*B11, A11*B12.

Each product appears in exactly one output. So no sharing is possible, and
the rank is 12.

OK so 2x2@2x3 has rank 12. My total would be 14 + 2*12 = 38.

The previous program gets 36 by a different decomposition. Let me look at it again.

Previous: 
- C1 = A B1 (2x2): split A into A1(2x2), A2(2x2). A1@B1_top(2x2) + A2@B1_bot(2x2).
  Two Strassen: 7+7=14.
- C2 = A B2 (2x3): split B2 into B21(4x1) and B22(4x2).
  C21 = A B21 (2x1): split A into A1,A2. A1@B21_top(2x1) + A2@B21_bot(2x1).
  Each 2x2@2x1 has rank 4. Total: 4+4=8.
  C22 = A B22 (2x2): split A into A1,A2. A1@B22_top(2x2) + A2@B22_bot(2x2).
  Two Strassen: 7+7=14.
  Total for C2: 8+14=22.
Grand total: 14+22=36.

This is correct. Can we improve?

Idea: Instead of splitting B into B1(4x2) and B2(4x3), what if we overlap?
Or use a different factorization.

Another idea: Use the fact that n=2 to write the problem as:
C[0,k] = A[0,:] . B[:,k]
C[1,k] = A[1,:] . B[:,k]

Let u = A[0,:] (1x4) and v = A[1,:] (1x4).
C[0,k] = u . B[:,k], C[1,k] = v . B[:,k].

So we need to compute two linear functionals of each column of B.
For each column k, we compute u.B[:,k] and v.B[:,k].
This is a 2x4 @ 4x1 multiplication per column, which has rank 8 per column.
5 columns: 40. Not better.

But we can share across columns! The key products are A[i,j]*B[j,k].
For fixed (i,j), the products A[i,j]*B[j,k] for k=0..4 are 5 products.
We have 8 such (i,j) pairs (2 rows * 4 cols). Total 40 products.

The question is: can we group these into bilinear forms that each contribute
to multiple outputs?

Let's try: for each j (0..3), consider the row B[j,:] (1x5) and the
2-vector [A[0,j], A[1,j]] (2x1). The contribution to C from column j is:
C += [A[0,j], A[1,j]] outer B[j,:]  (2x5)

This is a 2x1 @ 1x5 multiplication, which is rank 10 (each of the 10 entries
is a product). But we can do:
C[0,k] += A[0,j]*B[j,k], C[1,k] += A[1,j]*B[j,k].
That's 10 products per j, 4 j's, total 40. No saving.

I think 36 might be close to optimal for this small case. Let me verify
the previous program's decomposition is correct and just reproduce it cleanly.

Actually, let me try one more idea: can we do 2x4 @ 4x2 in less than 14?
2x4 @ 4x2: output 2x2, each entry is sum of 4 products. 16 products total.
Strassen-based gives 14. Can we do 13 or less?

The tensor rank of 2x4 @ 4x2 multiplication... this is equivalent to
(2x4) @ (4x2) = 2x2. We can view it as (2x4) @ (4x1) done twice (for the
two columns of the 4x2 matrix), but sharing the left-multiplication.

Actually, let's think of it as: C = A @ B where A is 2x4, B is 4x2.
C[i,k] = sum_j A[i,j] B[j,k].
Products: A[i,j]*B[j,k] for i in {0,1}, j in {0..3}, k in {0,1}.
That's 2*4*2 = 16 products, each in exactly one output C[i,k].
So rank is 16? No, because Strassen gives 14 < 16.

Wait, I'm confusing myself. In Strassen, the products are not the individual
A[i,j]*B[j,k] terms. They are combinations. For example, (A00+A11)*(B00+B11)
is one product, and it contributes to multiple outputs. So the rank counts
the number of such combined products, not the 16 individual ones.

For 2x4 @ 4x2, we can write A = [A1 A2] (two 2x2 blocks) and B = [B1; B2]
(two 2x2 blocks stacked). Then C = A1@B1 + A2@B2, each a 2x2@2x2.
Using Strassen for each: 7+7=14. Can we share between the two Strassen
computations? The products in A1@B1 involve A1 entries and B1 entries,
while A2@B2 involves A2 and B2. No overlap, so 14 is the rank via this
decomposition.

Could there be a direct rank-13 or less decomposition of 2x4@4x2?
I believe the tensor rank of (2,4,2) matrix multiplication is 14 (same as
two independent 2x2 multiplications since there's no sharing possible).

Actually, that's not quite right either. Let me just go with the known
decomposition that gives 36 and make sure it's correct.

Let me just carefully implement the 36-rank decomposition.
"""

import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    R = 36
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(k, i):
        return k * n + i

    col = [0]

    def strassen(ai, aj, bi, bk, ci, ck):
        c = col[0]
        # a: A[ai:ai+2, aj:aj+2], b: B[bi:bi+2, bk:bk+2]
        a00, a01 = uidx(ai, aj), uidx(ai, aj+1)
        a10, a11 = uidx(ai+1, aj), uidx(ai+1, aj+1)
        b00, b01 = vidx(bi, bk), vidx(bi, bk+1)
        b10, b11 = vidx(bi+1, bk), vidx(bi+1, bk+1)
        c00, c01 = widx(ck, ci), widx(ck+1, ci)
        c10, c11 = widx(ck, ci+1), widx(ck+1, ci+1)

        # M1 = (a00+a11)(b00+b11)
        U[a00, c] += 1; U[a11, c] += 1
        V[b00, c] += 1; V[b11, c] += 1
        W[c00, c] += 1; W[c11, c] += 1
        col[0] += 1

        # M2 = (a10+a11)b00
        U[a10, c] += 1; U[a11, c] += 1
        V[b00, c] += 1
        W[c10, c] += 1; W[c11, c] -= 1
        col[0] += 1

        # M3 = a00(b01-b11)
        U[a00, c] += 1
        V[b01, c] += 1; V[b11, c] -= 1
        W[c01, c] += 1; W[c11, c] += 1
        col[0] += 1

        # M4 = a11(b10-b00)
        U[a11, c] += 1
        V[b10, c] += 1; V[b00, c] -= 1
        W[c00, c] += 1; W[c10, c] += 1
        col[0] += 1

        # M5 = (a00+a01)b11
        U[a00, c] += 1; U[a01, c] += 1
        V[b11, c] += 1
        W[c00, c] -= 1; W[c01, c] += 1
        col[0] += 1

        # M6 = (a10-a00)(b00+b01)
        U[a10, c] += 1; U[a00, c] -= 1
        V[b00, c] += 1; V[b01, c] += 1
        W[c11, c] += 1
        col[0] += 1

        # M7 = (a01-a11)(b10+b11)
        U[a01, c] += 1; U[a11, c] -= 1
        V[b10, c] += 1; V[b11, c] += 1
        W[c00, c] += 1
        col[0] += 1

    def schoolbook_2x2_x_2x1(ai, aj, bi, bk, ci, ck):
        c = col[0]
        a00, a01 = uidx(ai, aj), uidx(ai, aj+1)
        a10, a11 = uidx(ai+1, aj), uidx(ai+1, aj+1)
        b0 = vidx(bi, bk)
        b1 = vidx(bi+1, bk)
        c0 = widx(ck, ci)
        c1 = widx(ck, ci+1)

        U[a00, c] = 1; V[b0, c] = 1; W[c0, c] = 1
        col[0] += 1
        U[a01, c] = 1; V[b1, c] = 1; W[c0, c] = 1
        col[0] += 1
        U[a10, c] = 1; V[b0, c] = 1; W[c1, c] = 1
        col[0] += 1
        U[a11, c] = 1; V[b1, c] = 1; W[c1, c] = 1
        col[0] += 1

    # C = A(2x4) @ B(4x5)
    # Split B into B1(4x2, cols 0-1) and B2(4x3, cols 2-4)
    # C1 = A @ B1 (2x2): Strassen on A1(2x2)@B1t(2x2) + A2(2x2)@B1b(2x2)
    strassen(0, 0, 0, 0, 0, 0)   # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    strassen(0, 2, 2, 0, 0, 0)   # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]

    # C2 = A @ B2 (2x3)
    # Split B2 into B2a(4x1, col 2) and B2b(4x2, cols 3-4)
    # C2a = A @ B2a (2x1): schoolbook A1(2x2)@B2at(2x1) + A2(2x2)@B2ab(2x1)
    schoolbook_2x2_x_2x1(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] @ B[0:2,2] -> C[0:2,2]
    schoolbook_2x2_x_2x1(0, 2, 2, 2, 0, 2)  # A[0:2,2:4] @ B[2:4,2] -> C[0:2,2]

    # C2b = A @ B2b (2x2): Strassen A1(2x2)@B2bt(2x2) + A2(2x2)@B2bb(2x2)
    strassen(0, 0, 0, 3, 0, 3)   # A[0:2,0:2] @ B[0:2,3:5] -> C[0:2,3:5]
    strassen(0, 2, 2, 3, 0, 3)   # A[0:2,2:4] @ B[2:4,3:5] -> C[0:2,3:5]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
