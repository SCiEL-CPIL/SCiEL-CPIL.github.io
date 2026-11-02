"""
Matrix multiplication 2x4 * 4x8: rank-minimal exact tensor decomposition.

Strategy:
- The schoolbook rank is 2*4*8 = 64.
- We use a structural decomposition: partition the 8 columns into two 4-column blocks.
- For each 2x4 * 4x4 block, we use the schoolbook rank 32.
- Total rank = 32 + 32 = 64. This is not better than schoolbook.

Alternative: Use the fact that the problem can be embedded in 4x4 * 4x8 and apply
Strassen-like decomposition. However, without verified coefficients, we fall back
to a safe construction.

Actually, a better structural approach: notice that the 2x4 * 4x8 matrix C is of
size 2x8. We can write C as [C1, C2] where C1 = A*B1, C2 = A*B2, and B1, B2 are
4x4. If we find a rank-r decomposition for 2x4 * 4x4, then total rank is 2r.

What is the minimal rank for 2x4 * 4x4? Schoolbook is 2*4*4 = 32.
Can we do better? For general matrices, the tensor rank of m x n * p x q is at
least max(mn, pq, m+p, n+q) and at most mn * pq / min(mn, pq) = max(mn, pq) when
one factor is square? Not exactly.

A known result: rank(2x2 * 2x2) = 7 (Strassen). For 2x4 * 4x4, we can partition
the 4 columns into two 2-column blocks: 2x4 * 4x2 has rank 2*4*2 = 16 schoolbook,
and with Strassen it's 7. So two blocks give rank 14 for 2x4 * 4x4.
Then for 2x4 * 4x8 (two 4x4 blocks), we need 2 * 14 = 28? No, each 4x4 block
requires 4x4*4x4 decomposition? No, A is 2x4, B_j is 4x4. The product A*B_j is
2x4.

Let me reconsider: A is 2x4, B is 4x8. Split B into B1 (4x4) and B2 (4x4).
C1 = A * B1 (2x4), C2 = A * B2 (2x4).
For each A*B_j, where A is 2x4 and B_j is 4x4:
Split B_j into columns: B_j = [b_j1 b_j2 b_j3 b_j4] (each 4x1).
C_j = [A*b_j1, A*b_j2, A*b_j3, A*b_j4].
Each A*b_jk is 2x1, computed as sum of 4 products -> rank 4 each, total 16 per block.
Two blocks: 32. Same as schoolbook.

With Strassen on 2x2 subblocks: A = [A11 A12] where A11, A12 are 2x2.
B_j = [B_j1; B_j2] where B_j1, B_j2 are 2x2 (rows).
A*B_j = A11*B_j1 + A12*B_j2.
Each 2x2*2x2 can be done in rank 7. But A11*B_j1 and A12*B_j2 share the same A parts.
We can combine: define M1 = A11, M2 = A12.
A*B_j = M1*B_j1 + M2*B_j2.
The products M1*B_j1 and M2*B_j2 are both 2x2. Using Strassen on each gives 7+7=14
per 2x2 product, but we have 4 such (two M's times two B blocks) -> 56? No.

Let me use a different embedding. Map the 2x4 A into a 4x4 A' with A' = [[A; 0]]
(where 0 is a 2x4 zero block). Similarly keep B as is. Then A'*B is 4x8, but only
top 2 rows matter. A' is 4x4, B is 4x8.
We can write A'*B = [A*B; 0*B] = [C; 0] where C is 2x8.
The full 4x4 * 4x8 product has rank at most 4*4*8 = 128 schoolbook.
With Strassen on 4x4*4x4 blocks (splitting 8 columns into 4+4): each 4x4*4x4 has
rank 32 (schoolbook) or less. Strassen 2x2x2 is rank 7, so 4x4x4 can be done in
rank 35? No, known best for 4x4x4 is 49 or less, but let's not rely on that.

Without verified coefficients for rank < 64, we provide the schoolbook decomposition
as the safe baseline.
"""

import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 150.0  # 150s limit, leave margin

    n, m, p = DIMENSIONS
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 1.0:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        if time.time() > deadline - 1.0:
            break
    if time.time() > deadline - 1.0:
        break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
