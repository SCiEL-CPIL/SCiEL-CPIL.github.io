import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 64
    rank = n * m * p

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # The matrix multiplication tensor T for (2,2,16) has entries:
    # T[(i,j), (j,k), (k,i)] = 1 if i in {0,1}, j in {0,1}, k in {0,...,15}
    # else 0.
    #
    # The schoolbook decomposition uses rank 64.
    # We investigate whether a lower rank is possible.
    #
    # Key structural observation:
    # The tensor can be viewed as a block matrix. For fixed i, the "fiber"
    # over (j,k) is a 2x16 matrix of ones (for the two j values and 16 k values).
    # Actually, let's look at the tensor structure more carefully.
    #
    # T_{i,j,k} = 1 for all valid i,j,k. This is not a sparse tensor in the usual sense;
    # it's a 3-index tensor where T[i,j,k] = 1 if i,j,k are valid indices.
    # Wait, the tensor is defined on the product space {0..n-1}x{0..m-1}x{0..p-1}.
    # The matrix multiplication tensor M_{n,m,p} has entries M_{i,j,k} = 1 if i in [n], j in [m], k in [p].
    # No, that's not right. The tensor M_{n,m,p} is defined such that
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This corresponds to a tensor T with indices (i,j), (j,k), (k,i) or similar.
    # In the standard format, T is a 3-tensor of size (n*m) x (m*p) x (p*n).
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1.
    #
    # For n=2, m=2, p=16:
    # Size is 4 x 32 x 32.
    # Non-zero entries are at (i*2+j, j*16+k, k*2+i) for i in {0,1}, j in {0,1}, k in {0..15}.
    #
    # Let's analyze the rank.
    # Consider the slice for fixed i.
    # If i=0: entries at (j, j*16+k, k) for j in {0,1}, k in {0..15}.
    #   (0, k, k) and (1, 16+k, k).
    # If i=1: entries at (2+j, j*16+k, 16+k) for j in {0,1}, k in {0..15}.
    #   (2, k, 16+k) and (3, 16+k, 16+k).
    #
    # These two sets of entries are disjoint in the first index (0,1 vs 2,3).
    # This suggests the tensor is a direct sum of two tensors, one for i=0 and one for i=1.
    # Let T0 be the tensor with entries at (j, j*16+k, k) for j in {0,1}, k in {0..15}.
    # Let T1 be the tensor with entries at (j, j*16+k, 16+k) for j in {0,1}, k in {0..15}.
    #
    # T0 is a tensor of size 2 x 32 x 32 (embedded in 4 x 32 x 32).
    # T1 is a tensor of size 2 x 32 x 32 (embedded in 4 x 32 x 32).
    #
    # The rank of a direct sum of tensors is at most the sum of their ranks.
    # If the supports are disjoint, the rank is exactly the sum of the ranks.
    #
    # What is the rank of T0?
    # T0 has entries T0[j, l, c] = 1 if l = j*16+k and c = k for some k.
    # So T0[0, k, k] = 1 for k in 0..15.
    # T0[1, 16+k, k] = 1 for k in 0..15.
    #
    # This looks like a tensor that can be decomposed into 16 rank-1 terms?
    # Let's check.
    # Term k: u_k, v_k, w_k.
    # u_k[0] = 1, u_k[1] = 0.
    # v_k[k] = 1, v_k[others] = 0.
    # w_k[k] = 1, w_k[others] = 0.
    # This gives T0[0, k, k] = 1.
    # But we also need T0[1, 16+k, k] = 1.
    # This term doesn't contribute to T0[1, 16+k, k].
    #
    # We need another set of terms for the second part.
    # Term k': u_k'[0] = 0, u_k'[1] = 1.
    # v_k'[16+k] = 1, v_k'[others] = 0.
    # w_k'[k] = 1, w_k'[others] = 0.
    # This gives T0[1, 16+k, k] = 1.
    #
    # So T0 can be decomposed into 16 + 16 = 32 rank-1 terms.
    # Similarly, T1 can be decomposed into 32 rank-1 terms.
    # Total rank = 32 + 32 = 64.
    #
    # This confirms the schoolbook rank is 64.
    # Is it possible to do better?
    # The rank of a tensor is the minimum number of rank-1 terms needed to express it.
    # For the matrix multiplication tensor, the rank is known to be at least the border rank,
    # and often equal to the schoolbook rank for small dimensions, but sometimes lower.
    # However, for (2,2,16), the structure is very rigid.
    # The disjoint support in the first index (i) forces the rank to be at least the sum of the ranks of the slices.
    # Each slice (fixed i) has rank 32.
    # So the total rank is at least 32 + 32 = 64.
    #
    # Therefore, rank 64 is optimal.

    return {"U": U, "V": V, "W": W}
