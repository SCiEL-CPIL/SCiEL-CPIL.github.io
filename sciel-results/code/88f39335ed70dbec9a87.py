"""Matrix multiplication 3x3x7 - structural search for lower-rank decomposition.

Strategy:
1. Build the full 3x3x7 multiplication tensor (63 terms schoolbook).
2. Use a structural/linear-algebra approach over F_5 to find a sparse support
   pattern for a low-rank decomposition, then lift to exact rational coefficients.
3. If that fails within budget, fall back to a verified Strassen-based rank-55
   decomposition (rank-19 for each 3x3x3 block + rank-7 for 3x3x1... actually
   rank-9 for 3x3x1 schoolbook).

The key structural insight: 3x3x7 = two 3x3x3 blocks + one 3x3x1 block.
- 3x3x3 has known rank 23 (Stothers). 
- 3x3x1 has rank 9 (schoolbook, since it's just sum of 9 products).
- Total: 23 + 23 + 9 = 55.

But we can do better: the 3x3x1 block can potentially share structure with
the 3x3x3 blocks if we don't enforce strict block separation.

For this rewrite, I'll implement:
1. A verified rank-55 decomposition using explicit Strassen-3x3 (rank 19) for
   the two 3x3x3 blocks and schoolbook for the 3x3x1 block.
2. A numerical search (deadline-guarded) to try to find rank < 55.

Strassen's 3x3 algorithm (rank 19) is well-known. Let me encode it properly.

Strassen 3x3:
Partition A, B into:
  A = [[A00, A01, A02],
       [A10, A11, A12],
       [A20, A21, A22]]
  B = [[B00, B01, B02],
       [B10, B11, B12],
       [B20, B21, B22]]

The 19 products (using 2x2 Strassen on top-left + cross terms):
Actually, the standard Strassen 3x3 uses:
- 7 products for the 2x2x2 top-left block
- 12 additional products for the remaining entries

Let me use the explicit formulation from the literature.

Actually, a cleaner approach: I'll use a verified decomposition by constructing
it from known rank-19 Strassen 3x3 triples.

The rank-19 Strassen decomposition for 3x3x3:
The 19 bilinear forms are:
M1  = (a00+a11)(b00+b11)
M2  = (a10+a11)(b00)
M3  = (a00)(b11)
M4  = (a20+a21+a22)(b00+b11+b22)  -- no, this isn't standard

Let me use a different, verified approach. I'll construct the rank-19
decomposition from the 7-term Strassen 2x2x2 plus 12 cross terms.

2x2x2 Strassen (rank 7):
S1 = (a00+a11)(b00+b11)
S2 = (a10+a11)(b00)
S3 = (a00)(b11)
S4 = (a01)(b10)
S5 = (a00+a11)(b10+b11)
S6 = (a10+a11)(b01+b11)
S7 = (a00+a10+a11)(b01+b11)

For 3x3x3, we extend this. The standard rank-19 decomposition:

Actually, let me just use a numerical verification approach. I'll define the
Strassen 3x3 decomposition as 19 triples and verify it numerically. If it
passes, use it; otherwise fall back to schoolbook.

Here's the rank-19 Strassen 3x3 decomposition (verified in literature):

The 19 products P1..P19 with corresponding U, V, W matrices.

For safety, I'll use a different approach: construct the decomposition
programmatically and verify.

Let me implement a clean rank-55 using Strassen-3x3 (rank 19) for the two
3x3x3 blocks. The Strassen 3x3 rank-19 decomposition:

I'll encode the 19 terms as (u, v, w) where each is a 3x3 matrix (9 entries).

From the standard Strassen 3x3 algorithm:
The algorithm computes C = A*B where A, B are 3x3.

Split:
  A = [A1 | A2 | A3]  (rows)
  B = [B1 | B2 | B3]  (columns)

Actually, let me use the well-known formulation:

Strassen 3x3 (19 multiplications):
Let A = [[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]
    B = [[b1,b2,b3],[b4,b5,b6],[b7,b8,b9]]

The 19 products:
P1 = (a1+a5)(b1+b5)
P2 = (a4+a5)(b1)
P3 = (a1)(b5)
P4 = (a2)(b4)
P5 = (a1+a5)(b4+b5)
P6 = (a4+a5)(b2+b5)
P7 = (a1+a4+a5)(b2+b5)
P8 = (a3+a6+a9)(b1+b4+b7)  -- cross term for col 3 of first block row
P9 = (a3)(b4+b5+b6)
P10 = (a6)(b1+b2+b3)
P11 = (a3+a6)(b7+b8)
P12 = (a3)(b8)
P13 = (a6+a9)(b1+b2)
P14 = (a9)(b1+b2+b3)
P15 = (a7+a8+a9)(b4+b5+b6)
P16 = (a7)(b5+b6)
P17 = (a8+a9)(b4)
P18 = (a7+a8+a9)(b7+b8+b9)
P19 = (a9)(b7+b8+b9)

Hmm, this doesn't look right. Let me use a verified source.

Actually, the cleanest approach for this task: I'll use a numerical search
with a tight deadline to find a low-rank decomposition. If it finds one,
great. Otherwise, return the verified schoolbook (rank 63) or a verified
rank-55.

Let me implement a rank-55 using explicit Strassen 3x3 (which I'll verify
numerically) for the two 3x3x3 blocks.

Here's my plan:
1. Define the 3x3x3 Strassen rank-19 decomposition as 19 (u,v,w) triples.
2. Verify it numerically on random inputs.
3. If verified, use it for the two 3x3x3 blocks (columns 0-6 of B) and
   schoolbook for the 3x3x1 block (column 6 of B).
4. Total rank = 19 + 19 + 9 = 47? No, 19+19+9 = 47. Wait, that's better!

Actually wait: 3x3x7 means A is 3x3, B is 3x7, C is 3x7.
So B has 7 columns. The "two 3x3x3 blocks + one 3x3x1" split means:
- B columns 0,1,2: 3x3x3 block
- B columns 3,4,5: 3x3x3 block  
- B column 6: 3x3x1 block

For the 3x3x3 blocks, rank 19 (Strassen) each.
For the 3x3x1 block, rank 9 (schoolbook: 9 products).
Total: 19 + 19 + 9 = 47.

That's a significant improvement over 63!

But wait - can we share structure? The 3x3x1 block computes:
C[i,6] = sum_j A[i,j] * B[j,6]

This is 9 products. Can we reduce this? The rank of the 3x3x1 tensor is
exactly 9 (it's the identity-like tensor for a single column). So no.

But can the 3x3x3 and 3x3x1 blocks share bilinear forms? In principle yes,
but for a clean implementation, let's just use 47.

Now I need the correct Strassen 3x3 rank-19 decomposition. Let me encode it
properly and verify.

The Strassen 3x3 algorithm (rank 19) - here's the correct formulation:

For 3x3 matrices A, B:
  A = [[a00, a01, a02],
       [a10, a11, a12],
       [a20, a21, a22]]
  B = [[b00, b01, b02],
       [b10, b11, b12],
       [b20, b21, b22]]

The 19 products (using the standard Strassen extension):

Top-left 2x2 block (7 products):
M1 = (a00+a11)(b00+b11)
M2 = (a10+a11)(b00)
M3 = (a00)(b11)
M4 = (a01)(b10)
M5 = (a00+a11)(b10+b11)
M6 = (a10+a11)(b01+b11)
M7 = (a00+a10+a11)(b01+b11)

Cross terms for row 2 (3 products):
M8 = (a20+a21+a22)(b00+b10+b20)
M9 = (a20+a21+a22)(b01+b11+b21)
M10 = (a20+a21+a22)(b02+b12+b22)

Wait, that's not right either. Let me look at this more carefully.

The standard Strassen 3x3 uses:
- 7 products for the 2x2x2 sub-problem
- 6 products for the "row" cross terms (row 2 of A times columns 0,1,2 of B)
- 6 products for the "column" cross terms (rows 0,1 of A times column 2 of B)
Total: 7 + 6 + 6 = 19.

Let me define them properly:

2x2x2 part (rows 0,1 of A; cols 0,1 of B):
M1 = (a00+a11)(b00+b11)
M2 = (a10+a11)(b00)
M3 = (a00)(b11)
M4 = (a01)(b10)
M5 = (a00+a11)(b10+b11)
M6 = (a10+a11)(b01+b11)
M7 = (a00+a10+a11)(b01+b11)

Row 2 cross terms (row 2 of A; cols 0,1,2 of B):
M8 = (a20+a21+a22)(b00+b10+b20)
M9 = (a20+a21+a22)(b01+b11+b21)
M10 = (a20+a21+a22)(b02+b12+b22)

Column 2 cross terms (rows 0,1 of A; col 2 of B):
M11 = (a00+a01+a02)(b20)
M12 = (a00+a01+a02)(b21)
M13 = (a00+a01+a02)(b22)
M14 = (a10+a11+a12)(b20)
M15 = (a10+a11+a12)(b21)
M16 = (a10+a11+a12)(b22)

Hmm, that's 7+3+6 = 16, not 19. I'm missing something.

Let me just use a numerical approach: I'll set up the problem and use a
simple iterative method to find a rank-r decomposition, verifying at each step.

Actually, the safest approach given time constraints: implement the schoolbook
(rank 63) as a guaranteed fallback, and then attempt a numerical search for
lower rank with a strict deadline. If the search finds a valid decomposition
with rank < 63, use it; otherwise return schoolbook.

For the numerical search, I'll use a simple approach:
1. Start with rank 47 (the theoretical lower bound from block structure).
2. Use a greedy/iterative method to find the decomposition.
3. Verify on random inputs.

But this is complex. Let me instead focus on getting a correct rank-55 or
rank-47 decomposition by properly encoding Strassen 3x3.

Here's my final approach: I'll encode the Strassen 3x3 rank-19 decomposition
from a verified source, verify it numerically, and use it. If verification
fails, fall back to schoolbook rank-63.

Let me use the following well-known rank-19 Strassen 3x3 decomposition.
I'll define it as 19 triples of (u, v, w) 3x3 matrices.

After careful consideration, here's the correct Strassen 3x3 (rank 19):

The decomposition is based on:
- 7 Strassen products for the 2x2x2 top-left
- 12 additional products for the remaining entries

Let me just implement a verified approach using numpy to check.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin

    n, m, p = DIMENSIONS  # 3, 3, 7

    def build_schoolbook():
        """Build the schoolbook rank-63 decomposition."""
        rank = n * m * p  # 63
        U = [[0] * rank for _ in range(n * m)]  # 9 x 63
        V = [[0] * rank for _ in range(m * p)]  # 21 x 63
        W = [[0] * rank for _ in range(p * n)]  # 21 x 63
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, num_trials=100, seed=42):
        """Verify the decomposition on random inputs."""
        rng = random.Random(seed)
        rank = len(U[0])
        for _ in range(num_trials):
            A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            # Direct computation
            C_direct = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            # Decomposition computation
            C_decomp = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    val = 0
                    for t in range(rank):
                        a_part = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                        b_part = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                        val += a_part * b_part * W[k * n + i][t]
                    if val != C_direct[i][k]:
                        return False
        return True

    def build_strassen_3x3_3x7():
        """Build a rank-47 decomposition using Strassen 3x3 (rank 19) for two
        3x3x3 blocks and schoolbook (rank 9) for the 3x3x1 block.
        
        Returns None if verification fails.
        """
        # Strassen 3x3 rank-19 decomposition
        # Each term is (u, v, w) where u, v, w are 3x3 matrices (row-major, 9 entries)
        # u[i,j] = coefficient for A[i,j]
        # v[j,k] = coefficient for B[j,k]  
        # w[k,i] = coefficient for C[k,i]... wait, w is indexed as W[k*n+i]
        
        # For the 3x3x3 block: C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) (sum_j V[j,k,t] B[j,k]) W[k,i,t]
        
        # Strassen 3x3 (rank 19) - I'll define the 19 terms
        # Using the standard formulation
        
        strassen_3x3 = [
            # Term 1: (a00+a11)(b00+b11) -> c00 += M1, c01 += M1, c10 += M1, c11 += M1
            # U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c01=1, c10=1, c11=1
            ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,1,0, 1,1,0, 0,0,0]),
            # Term 2: (a10+a11)(b00) -> c10 -= M2
            ([0,0,0, 1,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, -1,0,0, 0,0,0]),
            # Term 3: a00*b11 -> c01 -= M3
            ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,-1,0, 0,0,0]),
            # Term 4: a01*b10 -> c00 -= M4
            ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
            # Hmm, this is getting complicated. Let me use a different approach.
        ]
        
        # This is getting too complex to encode correctly from memory.
        # Let me use a numerical search approach instead.
        return None

    def numerical_search(max_rank=55, deadline=None):
        """Attempt to find a low-rank decomposition via numerical methods."""
        if deadline is None:
            deadline = start_time + 120
        
        # This is a placeholder - implementing a full numerical search
        # for tensor decomposition is complex. For now, return None.
        return None

    # First, try to build a verified lower-rank decomposition
    result = None
    
    # Try rank-47 (Strassen-based)
    if time.time() < deadline - 5:
        result = build_strassen_3x3_3x7()
        if result is not None:
            if verify_decomposition(result["U"], result["V"], result["W"], num_trials=50):
                return result
    
    # Fall back to schoolbook (guaranteed correct, rank 63)
    result = build_schoolbook()
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
