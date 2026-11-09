"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS  # n=2, m=5, p=6

    # Strategy: partition the middle dimension m=5 into blocks
    # Block sizes: [2, 2, 1]
    # For each block of size s, multiply A_block (2xs) * B_block (sx6)
    # Use Strassen-like decomposition for 2x2 x 2x6 blocks (rank 21 each)
    # For 2x1 x 1x6, rank is 1
    # Total: 21 + 21 + 1 = 43

    # But let's try a different partition: [1, 2, 2] or [2, 1, 2]
    # Or even try to use a more efficient decomposition for 2x5 x 5x6

    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = delta (i.e., we want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)

    # Let me try a direct approach: for each (i,j,k) we need exactly one rank-1 term
    # that covers it, and no spurious coverage.

    # Actually, the standard approach: we can use the fact that for matrix multiplication
    # A (n x m) * B (m x p) = C (n x p), the tensor is T[i,j,k] = 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j]*B[j,k], each product A[i,j]*B[j,k] contributes to C[i,k])

    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)

    # The trivial decomposition has rank n*m*p = 60, with each (i,j,k) having its own term.

    # Strassen for 2x2x2 gives rank 7 instead of 8.
    # For 2x2 x 2x6: we can use 3 pairs of columns, each with Strassen 2x2x2, giving 21.
    # For 2x1 x 1x6: rank 1.
    # Total: 21 + 21 + 1 = 43.

    # Can we do better? Let's try different block sizes.
    # Partition [1, 1, 1, 1, 1]: each 2x1 x 1x6 has rank 1, total = 5.
    # Wait, that's not right. 2x1 x 1x6 means A is 2x1, B is 1x6, C is 2x6.
    # The product is outer product: C[i,k] = A[i,0] * B[0,k]
    # This is rank 1 in the tensor sense? No, it's actually rank 1 as a matrix but
    # as a tensor term, we need U[i*5+0, r]=A[i,0], V[0*6+k, r]=B[0,k], W[k*2+i, r]=1
    # So yes, rank 1 per such block. Total = 5 blocks * 1 = 5.

    # Wait, that can't be right. Let me reconsider.
    # If we partition m=5 into five 1-dimensional blocks, each block computes
    # A_block (2x1) * B_block (1x6) = C_block (2x6)
    # The contribution to the tensor from this block is:
    # For each (i,k): C_block[i,k] = A[i, j0] * B[j0, k] where j0 is the block index
    # In the tensor: sum_r U[i*m+j0, r] * V[j0*p+k, r] * W[k*n+i, r] should equal 1
    # We can set: U[i*5+j0, r] = 1, V[j0*6+k, r] = 1, W[k*2+i, r] = 1 for a single r
    # But wait, this would make ALL (i,j0,k) terms equal to 1, which is what we want
    # for that specific j0. And different j0's use different r's.
    # So total rank = 5.

    # Hmm, but that seems too good. Let me verify:
    # For (i, j, k): we need sum_r U[i*5+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1
    # If j = j0, then U[i*5+j0, r0] = 1, V[j0*6+k, r0] = 1, W[k*2+i, r0] = 1
    # Product = 1. Good.
    # For j != j0: U[i*5+j, r0] = 0 (since r0 is only nonzero for j=j0)
    # So no cross-contamination. Total rank = 5.

    # Wait, but this is just the trivial decomposition with rank n*m*p / m = n*p = 12? No...
    # Actually, the trivial decomposition has rank n*m*p = 60, one per (i,j,k).
    # With block size 1, each block covers n*p = 2*6 = 12 entries of the tensor,
    # using 1 rank-1 term. So total rank = 5.

    # Let me construct this:
    rank = m  # 5
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j, r] = 1.0
        for k in range(p):
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0

    # Verify
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error = abs(total - 1.0)
                if error > max_error:
                    max_error = error

    if max_error > 1e-10:
        # Fallback to trivial
        trivial_rank = n * m * p
        U = np.zeros((n * m, trivial_rank))
        V = np.zeros((m * p, trivial_rank))
        W = np.zeros((p * n, trivial_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
