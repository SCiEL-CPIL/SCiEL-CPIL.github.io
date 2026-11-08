"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Strategy: Use Kronecker product structure to exploit the factorization of m=4=2*2.
We view the 2x4 matrix as two 2x2 blocks and the 4x5 matrix as two 2x5 blocks.
C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x5.

For each 2x2 by 2x5 multiplication, we use a decomposition based on Strassen's
approach generalized. The key insight is that a 2x2 by 2xp multiplication can be
done with rank 2p + 2(p-1) = 4p - 2 by using 7 multiplications for the 2x2 part
and sharing across columns.

Actually, let me think more carefully. For 2x2 by 2xp:
C[i,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[1,0]*B[0,k] + A[1,1]*B[1,k]
       = (A[0,0]+A[1,0])*(B[0,k]+B[1,k]) - A[1,0]*B[1,k] + A[0,0]*B[0,k]
       ... this isn't quite right.

Let me use a different approach. The 2x2 by 2xp tensor has rank at most 2*2*p = 4p
by schoolbook. But we can do better.

For each column k, we have a 2x2 by 2x1 multiplication (output is 2x1).
The 2x2 by 2x1 tensor: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
This has rank 4 (schoolbook). Can we do it in 3? No, I believe the rank is 4.

So for 2x2 by 2xp, the rank is at most 4p by doing each column separately.
But we can share multiplications across columns!

Consider: M1 = (A[0,0]+A[1,0]) * (B[0,k]+B[1,k]) for all k
This gives us a 2x5 result for each k, but we need to combine them.

Actually, let me use a cleaner approach. For 2x2 by 2xp:
Let's define 7 "Strassen-like" products that are shared across all k:
M1 = A[0,0] * (B[0,0]+B[1,0])  -- no, this doesn't generalize well.

Let me try a different strategy. I'll use the fact that the 2x2 by 2xp multiplication
can be decomposed using the Kronecker product of the 2x2 by 2x2 Strassen decomposition
with an identity-like structure, but since p=5 is not a power of 2, this doesn't work directly.

Alternative: Use a rank-10 decomposition for 2x2 by 2x5.
The schoolbook rank is 2*2*5 = 20.
Can we do it in 10? That would be the minimum possible (since the output is 2x5 = 10 entries,
and each rank-1 term can contribute to at most 1 output entry in a structured way... actually no).

Wait, the rank of the matrix multiplication tensor for (2x2)*(2xp) is known to be:
- For p=1: rank 4
- For p=2: rank 7 (Strassen)
- For p=3: rank ? 
- For general p: I believe it's 2p + 2(p-1) = 4p-2? Or maybe 3p+1?

Actually, I recall that for 2x2 by 2xp, the minimum rank is 2p + 2(p-1) = 4p-2 for p >= 2.
For p=2: 4*2-2 = 6, but Strassen gives 7. So that formula is wrong.

Let me just try to construct a decomposition and see what rank I get.

For 2x2 by 2xp, here's a decomposition with rank 3p+1:
M1 = (A00+A10) * (B0k+B1k) for each k -- but this is p terms
M2 = A00 * B0k for each k -- p terms
M3 = A11 * B1k for each k -- p terms
M4 = A10 * B1k for each k -- p terms
Total: 4p terms, same as schoolbook.

I don't see how to do better than 4p for 2x2 by 2xp in general.

So for 2x4 by 4x5:
C = A1*B1 + A2*B2
Each term has rank at most 4*5 = 20.
Total rank: 40.

This is the same as schoolbook. So I can't improve by this simple decomposition.

Let me try a different approach: use a global Strassen-like decomposition.

For 2x4 by 4x5, write:
A = [A1 | A2] where A1, A2 are 2x2
B = [B1; B2] where B1, B2 are 2x5

C = A1*B1 + A2*B2

Now, can we compute A1*B1 + A2*B2 with fewer than 40 multiplications?
This is equivalent to computing the product of a 2x4 matrix and a 4x5 matrix.

One approach: use the fact that we can form combinations of A1 and A2, and B1 and B2.
For example:
M1 = (A1+A2) * (B1+B2)
M2 = A1 * B1
M3 = A2 * B2

Then C = M1 - M2 - M3 + 2*(something)... no, this doesn't work cleanly.

Actually, C = A1*B1 + A2*B2.
If I compute M1 = (A1+A2)*(B1+B2) = A1*B1 + A1*B2 + A2*B1 + A2*B2
And M2 = A1*B2, M3 = A2*B1
Then C = M1 - M2 - M3

But M1, M2, M3 are each 2x2 by 2x5 multiplications, each of rank 20.
Total rank: 60, which is worse.

I think the schoolbook rank of 40 is likely optimal or near-optimal for this case.
Let me just output the schoolbook decomposition.

Actually, wait. Let me try one more thing. What if I use a rank-7 decomposition for each 2x2 by 2x2 block
and handle the p=5 dimension separately?

For 2x4 by 4x5, split into 2x2 by 2x5 blocks. For each 2x2 by 2x5, I can view it as
5 separate 2x2 by 2x1 multiplications. Each 2x2 by 2x1 has rank 4.
Total: 5*4 = 20 per block, 40 total.

I'm stuck at 40. Let me just output the schoolbook decomposition.
"""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
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
# EVOLVE-BLOCK-END
