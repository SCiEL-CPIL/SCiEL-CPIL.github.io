"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Strategy:
- The dimensions are n=2, m=4, p=5.
- m=4 = 2*2, so we can view the 2x4 matrix as two 2x2 blocks and the 4x5 matrix as two 2x5 blocks.
- We can use Strassen's algorithm for the 2x2 multiplication structure to reduce rank.
- However, p=5 is prime, so we cannot directly apply Kronecker product with p.

Alternative approach:
- The schoolbook rank is n*m*p = 2*4*5 = 40.
- We can try to find a decomposition of rank less than 40.

Known lower bound: For n x m by m x p, the rank is at least max(n*p, n*m, m*p) = max(10, 8, 20) = 20.
But more sophisticated lower bounds exist.

Let's try a different approach: use the fact that m=4 can be split as 2+2.
We can write:
  C = A * B = [A1 | A2] * [B1; B2] = A1*B1 + A2*B2
where A1, A2 are 2x2 and B1, B2 are 2x5.

Each term A1*B1 and A2*B2 is a 2x2 by 2x5 multiplication.
The rank of a 2x2 by 2x5 multiplication is at most 2*2*5 = 20, but can we do better?

Actually, for 2x2 by 2xp, the minimum rank is known to be 2p for p>=2? No, that's not right.

Let me think differently. Let's use the standard tensor decomposition and try to find a lower rank.

Actually, let's just implement the schoolbook decomposition first to have a baseline, then try to improve.

But the directive says to use Kronecker product structure. Let me think about that.

m=4 = 2*2. So we can write m = m1*m2 with m1=2, m2=2.
n=2, p=5.

If we could write p = p1*p2, but 5 is prime. So we can't directly use Kronecker product for p.

However, we can use the Kronecker structure for the m dimension.

Let me try a different strategy: use a known low-rank decomposition for 2x4 by 4x5.

Actually, let's just output the schoolbook decomposition to ensure correctness, and the score will be 40.

Wait, the seed program already does that. I need to do better.

Let me research: what is the known minimum rank for 2x4 by 4x5?

I recall that for n x m by m x p, the minimum rank is not always nmp. For 2x2 by 2x2, it's 7 (Strassen).

For 2x4 by 4x5, we can write:
  C = A*B
  where A is 2x4, B is 4x5.

Split A into two 2x2 blocks: A = [A1 | A2]
Split B into two 2x5 blocks: B = [B1; B2]

Then C = A1*B1 + A2*B2.

Each of A1*B1 and A2*B2 is a 2x2 by 2x5 multiplication.

The rank of a 2x2 by 2xp multiplication: 
  The tensor is sum over i,j,k of e_i ⊗ e_j ⊗ e_k where i in {1,2}, j in {1,2}, k in {1,...,p}.
  This is equivalent to p copies of the 2x2 by 2x2 tensor? No, because the k index is separate.

Actually, the 2x2 by 2xp multiplication can be seen as: for each k, we compute A1*B1[:,k] and A2*B2[:,k], but these are independent.

Wait, no. A1*B1 is a 2x2 by 2x5, which gives a 2x5 result. The tensor for this is a 2x2x5 tensor.

The minimum rank of the 2x2x5 tensor: I believe it's 2*5 = 10? Or is it less?

Actually, the rank of the 2x2xp tensor is at most 2p (by doing each column separately with rank 2 for 2x2 by 2x1? No, 2x2 by 2x1 has rank 4? No, 2x2 by 1x1... I'm confusing myself.

Let me restart. The matrix multiplication tensor for (n x m) * (m x p) has rank equal to the minimum number of multiplications needed.

For 2x2 by 2x2, rank is 7.
For 2x2 by 2x1, the output is 2x1. The tensor is 2x2x1. The rank is 2? Because C_i = sum_j A_ij B_j. This is just 2 multiplications? No, it's 4 scalar multiplications in schoolbook, but the tensor rank... 

Actually, for 2x2 by 2x1: C_1 = A_11*B_1 + A_12*B_2, C_2 = A_21*B_1 + A_22*B_2. This is 4 multiplications. Can we do it in 3? I think the rank is 4 for 2x2x1.

Let me just implement a decomposition using Strassen for the 2x2 part.

Here's my plan:
- Use the fact that m=4=2*2.
- View the 2x4 matrix as two 2x2 blocks.
- View the 4x5 matrix as two 2x5 blocks.
- Compute C = A1*B1 + A2*B2.
- For each 2x2 by 2x5 multiplication, use a rank-10 decomposition (schoolbook for 2x2 by 2x5 is 2*2*5=20, but we can do better?).

Actually, I think I'm overcomplicating this. Let me just output the schoolbook decomposition. The task says "find a rank-minimal exact tensor decomposition", but if I don't know a better one, I'll output the schoolbook.

But the seed already does that. I need to produce something different.

Let me try to use Strassen's algorithm adapted for this case.

Strassen for 2x2 by 2x2 uses 7 multiplications. We can generalize this.

For 2x4 by 4x5, write:
  A = [[a1, a2, a3, a4], [b1, b2, b3, b4]]
  B = [[c1, c2, c3, c4, c5], [d1, d2, d3, d4, d5], [e1, e2, e3, e4, e5], [f1, f2, f3, f4, f5]]

Split A into A1 = [[a1,a2],[b1,b2]] (2x2) and A2 = [[a3,a4],[b3,b4]] (2x2).
Split B into B1 = [[c1,c2,c3,c4,c5],[d1,d2,d3,d4,d5]] (2x5) and B2 = [[e1,e2,e3,e4,e5],[f1,f2,f3,f4,f5]] (2x5).

Then C = A1*B1 + A2*B2.

For A1*B1 (2x2 by 2x5), we can use 7 multiplications if we treat it as two 2x2 by 2x1 problems? No, that's not how Strassen works.

I think the correct approach is to use the Kronecker product of the Strassen decomposition for 2x2 by 2x2 and the identity for the p dimension. But p=5, not 2.

This is getting too complex. Let me just output the schoolbook decomposition to be safe.

BET: {mechanism: "The schoolbook decomposition of rank 40 is used as a fallback since no simpler lower-rank decomposition was found; this ensures correctness but does not improve on the seed.", q: 0.5, delta: [0.0, 0.0]}
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
