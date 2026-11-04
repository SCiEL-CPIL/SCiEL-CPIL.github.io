"""Strategy: Block-based rank-18 decomposition for 2x3 * 3x5 matrix multiplication.

Decompose the 3x3 multiplication core (indices 0..2 in A, 0..2 in B) using a rank-3
Strassen-like decomposition.  The remaining B columns (indices 3,4) are handled with
a rank-1 term each.  The A rows are n=2 rows, so the total rank is
2 * 3 (Strassen core) + 2 * 2 (extra columns) = 10.

Wait: that gives rank 10, but the task says rank is the score (lower is better).
Let me think more carefully.

Actually, let's use a simpler approach:
- For the 3x3 block (A rows 0,1; B cols 0,1,2), use rank-3 Strassen decomposition per A row.
  That's 2 * 3 = 6 rank-1 tensors.
- For B cols 3,4, use rank-1 per (A row, B col): 2 * 2 = 4 rank-1 tensors.
Total rank = 10.

But Strassen for 2x2 is rank 7. For 2x3 * 3x3, we can use a 2x2 Strassen on the first
2 columns of B, plus 1 extra column, giving 7 + 2 = 9 per... no, that's not right either.

Let me just use the known fact: rank(2x3, 3x5) is at most 2 * rank(3x3, 3x3) if we
embed, but that's not quite right.

Let me try a direct construction:
- Use the schoolbook for the 2x3 * 3x5: rank 30 (baseline).
- Try to find a lower rank by exploiting structure.

Actually, the simplest improvement: use a rank-7 Strassen for the 2x2 * 2x2 block
(first 2 rows of A, first 2 cols of B), plus schoolbook for the rest.

A is 2x3, B is 3x5.
Split B into B1 (3x2) and B2 (3x3).
C = A * B1 + A * B2.

For A * B1: A is 2x3, B1 is 3x2. We can use rank-7 Strassen-like? No, Strassen is for
square matrices. For 2x3 * 3x2, the tensor rank is at most 2*3*2 = 12 schoolbook.
Actually, I think the rank of 2x3 * 3x2 multiplication is 6 (same as 2x2 * 2x2 is 7,
but 2x3 * 3x2 might be different).

Let me just try to construct a rank-18 decomposition as the strategy suggests:
Use a rank-3 decomposition for the 3x3 core (columns 0,1,2 of B), with the 2 rows of A
handled separately. So 2 * 3 = 6 for the core, plus 2 * 2 = 4 for the extra columns
3,4. Total = 10.

But I need to verify this works. Let me just code up a rank-10 attempt and verify.
"""

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    deadline = time.time() + 120  # safety margin before 150s kill

    # Strategy: Build a rank-10 decomposition.
    # 
    # The multiplication tensor T[i, j, k] for i in [0,2), j in [0,3), k in [0,5)
    # is defined by: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) tuples.
    #
    # We need rank-1 tensors (u, v, w) such that:
    #   sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    #
    # Approach: 
    #   - For k in {0,1,2}: use a rank-3 decomposition of the 3x3 multiplication
    #     (treating A as 2x3, B-core as 3x3, but we need to handle the 2 rows).
    #     Actually, let's use a simpler block approach.
    #
    #   - For each (i, k) pair, we need: sum_j A[i,j] * B[j,k] = C[i,k]
    #     This is automatically satisfied by the schoolbook.
    #
    #   Let's try: use rank-1 tensors for each (i, j) pair, but group by k.
    #     For fixed i and j, we need to cover all k. That's p=5 tensors per (i,j).
    #     Total: 2 * 3 * 5 = 30. No improvement.
    #
    #   Better: Use the fact that for fixed j, the row A[i,j] multiplies the column
    #     B[j,:]. So for each j, we have a rank-1 outer product A[i,j] * B[j,k].
    #     But that's still schoolbook.
    #
    #   Let's try a rank-9 approach:
    #     - For B columns 0,1: use a 2x2 Strassen-like on the first 2 rows of A
    #       (which are all rows since n=2) and first 2 cols of B. Rank 7.
    #     - For B column 2: rank-1 per row of A: 2 tensors.
    #     - For B columns 3,4: rank-1 per row of A: 4 tensors.
    #     Total: 7 + 2 + 4 = 13.
    #
    #   Actually, for 2x2 * 2x2, Strassen has rank 7. Let's use that.
    #   A is 2x3, so we split A into A[:,0:2] (2x2) and A[:,2] (2x1).
    #   B is 3x5, so we split B into B[0:2,:] (2x5) and B[2,:] (1x5).
    #
    #   C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    #
    #   For A[:,0:2] @ B[0:2,:]: this is 2x2 * 2x2 * 5? No, it's (2x2) @ (2x5) = 2x5.
    #   We can view this as: for each of the 5 columns of B[0:2,:], we do a 2x2 * 2x1
    #   multiplication. That's 5 independent 2x2 * 2x1 multiplications, each rank at most 4.
    #   Total: 5 * 4 = 20. Worse than schoolbook for this block (5*4=20 vs 2*2*5=20). Same.
    #
    #   Alternatively, treat it as a single tensor: rank of (2x2) * (2x5) multiplication.
    #   This is the same as 2x2 * 2x1 tensor multiplied by 5 independent columns.
    #   The rank is 5 * rank(2x2 * 2x1) = 5 * 4 = 20? Or can we share?
    #   Actually, the tensor for (2x2) * (2x5) is a 2x2x5 tensor, which has rank
    #   at most 2*2*5 = 20. But it might be lower.
    #
    #   I think the safest bet is to just try to find a rank-18 or rank-10 decomposition
    #   by solving a system of equations.
    #
    #   Let's try a different approach: use the known rank of 2x3 * 3x5.
    #   According to the literature, rank(2x3, 3x5) is at most 2*3*5 = 30 schoolbook,
    #   but can be lower. I recall that rank(2x3, 3x3) is at most 9 (using a 3x3
    #   decomposition with rank 3 per row of A, giving 2*3=6? No, that's not right).
    #
    #   Let me just try to construct a rank-12 decomposition:
    #     - Use 2 independent rank-6 decompositions for the 2 rows of A.
    #     - For row i of A, we need to compute sum_j A[i,j] * B[j,k] for k=0..4.
    #     - This is a 1x3 * 3x5 multiplication, which has rank at most 15 schoolbook,
    #       but can be done in rank 5 (one per column of B).
    #     - So total rank = 2 * 5 = 10.
    #
    #   Wait, that's just the schoolbook again: 2*3*5 = 30. No, 2*5 = 10 means for
    #     each row i, we use 5 rank-1 tensors (one per k), each with A[i,:] * B[:,k].
    #     But A[i,:] is 1x3 and B[:,k] is 3x1, so the rank-1 tensor is:
    #       u_r[i,j] = A[i,j], v_r[j,k] = B[j,k], w_r[k,i] = 1
    #     But we can't use A and B directly because they're variable inputs.
    #     The decomposition must be fixed coefficients.
    #
    #   OK, I think I need to just use the schoolbook rank-30 and return that,
    #   since I can't quickly find a better construction.
    #
    #   Actually, let me try one more thing: the rank of 2x3 * 3x5 is at most
    #   2 * rank(3x3 * 3x5) if we can decompose the 3x3 part. But I don't know
    #   rank(3x3 * 3x5).
    #
    #   Let me just return the schoolbook decomposition with rank 30.
    
    rank = n * m * p  # 30
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_computed[i, j, k] = np.sum(
                    U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                )
    expected = np.ones((n, m, p))
    if not np.allclose(T_computed, expected):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
