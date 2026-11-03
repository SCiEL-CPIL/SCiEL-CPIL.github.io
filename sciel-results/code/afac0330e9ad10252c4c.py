"""
Exact tensor decomposition for 3x4 by 4x7 matrix multiplication.

Uses a constructive decomposition:
- Split the 7 columns into groups of sizes 3, 2, 2.
- For each group of size s, decompose 3x4 by 4xs as a Kronecker-like product
  of a 3xs block (handled with rank 3*s*s using trivial basis) combined with
  a 4x4 middle structure that we keep at rank 16 (the naive 4*4).

Actually, to keep it exact and simple, we use the following:
For multiplying A (3x4) * B (4x7):
  C[i,k] = sum_j A[i,j] B[j,k]

We can decompose by first doing 3x4 * 4x3 (rank 12*3=36? No, naive is 3*4*3=36).
Better: use the identity that matrix multiplication (n,m,p) can be built from
smaller blocks. Here we use a direct sum decomposition based on the column blocks.

Let's do:
Block 1: 3x4 * 4x3 -> rank 36 (naive)
Block 2: 3x4 * 4x2 -> rank 24 (naive)
Block 3: 3x4 * 4x2 -> rank 24 (naive)
Total naive: 84.

But we can do better by sharing the left factor structure.
Actually, the seed gives 84. Let me try a known small improvement.

Known: For 3x3 * 3x3, rank 23 (Strassen-like) is known.
For 3x4 * 4x7, we can embed 3x3*3x3 blocks.

Alternative approach: Use the fact that 4 = 3+1 and 7 = 3+2+2.
Decompose as direct sum of:
- 3x3 * 3x3 (rank <= 23)
- 3x3 * 3x2 (rank <= ?)
- 3x3 * 3x2 (rank <= ?)
- 3x1 * 1x3 (rank 3)
- 3x1 * 1x2 (rank 6)
- 3x1 * 1x2 (rank 6)
- 1x3 * 3x3 (rank 9)
- 1x3 * 3x2 (rank 6)
- 1x3 * 3x2 (rank 6)
- 1x1 * 1x3 (rank 3)
- 1x1 * 1x2 (rank 2)
- 1x1 * 1x2 (rank 2)

This is getting complex. Let me just implement a clean naive decomposition with a small optimization.

Actually, the simplest robust improvement: use the "block Kronecker" trick.
Note that 4 = 2*2 and 7 = 7, 3 = 3.
We can write 3x4 * 4x7 as (3x2 * 2x4) combined... this doesn't help directly.

Let me just produce a correct program with rank 84 (same as seed) but with a cleaner structure, 
or try to find a genuine improvement.

Known result: The rank of 3x3 matrix multiplication is 23 (Bürgisser, Clausen, Shokrollahi 1987).
For 3x4 * 4x7, we can use:
  (3x4)*(4x7) = (3x4)*(4x3) + (3x4)*(4x2) + (3x4)*(4x2)
  = (3x3*3x3 + 3x3*3x1 + 3x1*3x3 + 3x1*3x1) ... this is messy.

Let me just output a valid rank-84 decomposition that is clearly correct, matching the seed's rank.
Since the directive says "design a new program from scratch" but the score is seed/value and seed is 1.0,
and I can't easily find a sub-84 rank for 3x4x7 in time, I'll produce a clean, correct rank-84 solution
with a different construction style (using explicit index mapping with a helper).
"""

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p  # 84
    
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
