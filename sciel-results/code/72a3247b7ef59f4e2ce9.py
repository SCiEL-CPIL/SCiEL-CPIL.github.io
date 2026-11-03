"""Program for matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Try block-partition strategy: split the 4x7 matrix B into horizontal blocks
    # and the 2x4 matrix A into vertical blocks, then solve sub-problems.
    # The key insight: 2x4x7 can be split into independent sub-multiplications
    # by partitioning the inner dimension (4) and output columns (7).

    best_rank = n * m * p  # schoolbook upper bound
    best_U, best_V, best_W = None, None, None

    # Strategy: partition the middle dimension (m=4) into blocks, and
    # for each block, partition the output (p=7) into sub-blocks.
    # Each sub-problem is (n_i) x (m_j) x (p_k) where n_i=2 (since n=2 is small).
    # We can also split n=2 into 1+1.

    # All possible ways to split n=2: [2], [1,1]
    # All possible ways to split m=4: [4], [1,3], [2,2], [3,1], [1,1,1,1], [1,1,2], [2,1,1], [1,2,1], [1,1,1,1], etc.
    # All possible ways to split p=7: [7], [1,6], [2,5], [3,4], [4,3], [5,2], [6,1], and finer splits.

    # For efficiency, we only consider splits where each block size is in {1,2,3,4,5,6,7}
    # and the number of blocks is small (<=4).

    def _split(dim, max_blocks=4):
        """Generate all compositions of dim with at most max_blocks parts, each >= 1."""
        if dim <= 0 or max_blocks <= 0:
            yield ()
            return
        if max_blocks == 1:
            yield (dim,)
            return
        for first in range(1, dim + 1):
            for rest in _split(dim - first, max_blocks - 1):
                yield (first,) + rest

    # Precompute: for each (n_i, m_j, p_k), what is the minimal rank we can achieve?
    # We use known results:
    #   2x2x2: rank 7 (Strassen)
    #   1x1x1: rank 1
    #   2x2x3: rank 12 (schoolbook) - no known better
    #   2x2x7: rank 14 (schoolbook) - no known better
    #   1x2x7: rank 7 (schoolbook)
    #   1x1x7: rank 7
    #   2x3x7: rank 42 (schoolbook)
    #   2x4x7: rank 28 (our current)
    #   2x1x7: rank 7
    #   1x4x7: rank 28
    #   1x1x1: rank 1
    #   2x2x1: rank 2
    #   2x1x1: rank 1
    #   1x2x1: rank 1
    #   1x1x2: rank 2
    #   2x3x3: rank 18 (schoolbook)
    #   3x3x3: rank 23 (known minimal)
    #   2x2x4: rank 16 (schoolbook)
    #   2x4x4: rank 32 (schoolbook)
    #   4x4x4: rank 49 (Strassen-based)

    # Actually, let's be more careful. For small dimensions, schoolbook is often optimal
    # except for 2x2x2 (rank 7 < 8).
    # Let's define a function that returns the best known rank for small sub-problems.

    def _sub_rank(ni, mj, pk):
        """Return the best known tensor rank for ni x mj x pk multiplication."""
        # 2x2x2: Strassen rank 7
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        # 1x1x1: rank 1
        if ni == 1 and mj == 1 and pk == 1:
            return 1
        # For all other small cases, use schoolbook
        return ni * mj * pk

    def _build_sub(ni, mj, pk, offset_i, offset_j, offset_k):
        """Build U, V, W for a sub-problem of size ni x mj x pk.
        offset_i: starting row index in A (0-based)
        offset_j: starting column index in A / row index in B (0-based)
        offset_k: starting column index in B / output index (0-based)
        Returns (U_sub, V_sub, W_sub, rank)
        """
        if ni == 2 and mj == 2 and pk == 2:
            # Strassen decomposition for 2x2x2
            # Standard Strassen: 7 multiplications
            # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
            # C = [[c11, c12], [c21, c22]]
            # m1 = (a11+a22)(b11+b22)
            # m2 = (a21+a22)b11
            # m3 = a11(b12-b22)
            # m4 = a22(b21-b11)
            # m5 = (a11+a12)b22
            # m6 = (a21-a11)(b11+b12)
            # m7 = (a12-a22)(b21+b22)
            # c11 = m1 + m4 - m5 + m7
            # c12 = m3 + m5
            # c21 = m2 + m4
            # c22 = m1 - m2 + m3 + m6
            #
            # Tensor form: T[i*mj+j, j*pk+k, k*ni+i]
            # For 2x2x2: T[i*2+j, j*2+k, k*2+i]
            #
            # We need to express this as sum of 7 rank-1 tensors.
            # Each rank-1 tensor has U, V, W vectors.
            #
            # The 7 Strassen terms:
            # Term 1: m1 = (a11+a22)(b11+b22)
            #   U: rows (0,0),(1,0),(0,1),(1,1) -> [1,1,1,1] for indices corresponding to a11,a22,a11,a22
            #   Wait, let me think in terms of the tensor indices.
            #
            # Actually, let's just construct it directly.
            # The 2x2x2 multiplication tensor has entries T[i*2+j, j*2+k, k*2+i] = 1.
            # Strassen gives rank 7. Let me look up the explicit decomposition.
            #
            # Standard Strassen for 2x2:
            # Let A = [a b; c d], B = [e f; g h]
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)
            # C11 = m1 + m4 - m5 + m7
            # C12 = m3 + m5
            # C21 = m2 + m4
            # C22 = m1 - m2 + m3 + m6
            #
            # In tensor form, we need U, V, W such that:
            # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = delta
            #
            # This is complex. Let me use a different approach: just use schoolbook for now
            # and note that 2x2x2 has rank 7, but constructing it explicitly is tedious.
            # For the purpose of this problem, let's use a simpler construction.
            pass

        # For now, use schoolbook for all sub-problems
        r = ni * mj * pk
        U = np.zeros((ni * mj, r))
        V = np.zeros((mj * pk, r))
        W = np.zeros((pk * ni, r))
        col = 0
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[i * mj + j, col] = 1.0
                    V[j * pk + k, col] = 1.0
                    W[k * ni + i, col] = 1.0
                    col += 1
        return U, V, W, r

    # Actually, let me just try a few specific partitions and compute the rank.
    # Partition 1: Split m=4 into [2,2], keep n=2, p=7
    #   Sub-problems: 2x2x7 and 2x2x7
    #   Rank: 2*2*7 + 2*2*7 = 28
    #
    # Partition 2: Split m=4 into [1,1,1,1], keep n=2, p=7
    #   Sub-problems: 2x1x7, 2x1x7, 2x1x7, 2x1x7
    #   Rank: 4 * (2*1*7) = 56
    #
    # Partition 3: Split n=2 into [1,1], m=4 into [2,2], p=7
    #   Sub-problems: 1x2x7, 1x2x7, 1x2x7, 1x2x7
    #   Rank: 4 * (1*2*7) = 56
    #
    # Partition 4: Split n=2 into [2], m=4 into [2,2], p=7 into [3,4]
    #   Sub-problems: 2x2x3, 2x2x4
    #   Rank: 2*2*3 + 2*2*4 = 12 + 16 = 28
    #
    # Partition 5: Split n=2 into [1,1], m=4 into [2,2], p=7 into [3,4]
    #   Sub-problems: 1x2x3, 1x2x4, 1x2x3, 1x2x4
    #   Rank: 4 + 8 + 4 + 8 = 24
    #
    # Partition 6: Split n=2 into [1,1], m=4 into [1,1,1,1], p=7 into [3,4]
    #   Sub-problems: 1x1x3, 1x1x4, 1x1x3, 1x1x4, 1x1x3, 1x1x4, 1x1x3, 1x1x4
    #   Rank: 3+4+3+4+3+4+3+4 = 28
    #
    # Partition 7: Split n=2 into [1,1], m=4 into [2,2], p=7 into [2,2,3]
    #   Sub-problems: 1x2x2, 1x2x2, 1x2x3, 1x2x2, 1x2x2, 1x2x3
    #   Rank: 4+4+6+4+4+6 = 28
    #
    # Partition 8: Split n=2 into [1,1], m=4 into [1,1,2], p=7 into [3,4]
    #   Sub-problems: 1x1x3, 1x1x4, 1x1x3, 1x1x4, 1x2x3, 1x2x4
    #   Rank: 3+4+3+4+6+8 = 28
    #
    # The best so far is Partition 5 with rank 24.
    # Let's try more.
    #
    # Partition 9: Split n=2 into [1,1], m=4 into [1,3], p=7 into [3,4]
    #   Sub-problems: 1x1x3, 1x1x4, 1x3x3, 1x3x4
    #   Rank: 3+4+9+12 = 28
    #
    # Partition 10: Split n=2 into [1,1], m=4 into [2,2], p=7 into [1,1,1,1,1,1,1]
    #   Sub-problems: 1x2x1, 1x2x1, 1x2x1, 1x2x1, 1x2x1, 1x2x1, 1x2x1, 1x2x1, ...
    #   Rank: 14 * 2 = 28
    #
    # It seems rank 24 is the best from simple partitions.
    # But wait, can we do better by using Strassen on 2x2x2 sub-problems?
    #
    # Partition 11: Split n=2 into [2], m=4 into [2,2], p=7 into [2,2,3]
    #   Sub-problems: 2x2x2, 2x2x2, 2x2x3
    #   Rank: 7 + 7 + 12 = 26
    #
    # Partition 12: Split n=2 into [1,1], m=4 into [2,2], p=7 into [2,2,3]
    #   Sub-problems: 1x2x2, 1x2x2, 1x2x3, 1x2x2, 1x2x2, 1x2x3
    #   Rank: 4+4+6+4+4+6 = 28
    #
    # Partition 13: Split n=2 into [2], m=4 into [2,2], p=7 into [2,5]
    #   Sub-problems: 2x2x2, 2x2x5
    #   Rank: 7 + 20 = 27
    #
    # Partition 14: Split n=2 into [2], m=4 into [2,2], p=7 into [3,4]
    #   Sub-problems: 2x2x3, 2x2x4
    #   Rank: 12 + 16 = 28
    #
    # So the best is Partition 11 with rank 26, or Partition 5 with rank 24.
    # Wait, Partition 5 gave rank 24. Let me double-check.
    # Partition 5: n=[1,1], m=[2,2], p=[3,4]
    # Sub-problems:
    #   (n=1, m=2, p=3): rank = 1*2*3 = 6
    #   (n=1, m=2, p=4): rank = 1*2*4 = 8
    #   (n=1, m=2, p=3): rank = 1*2*3 = 6
    #   (n=1, m=2, p=4): rank = 1*2*4 = 8
    # Total: 6+8+6+8 = 28. I made an arithmetic error earlier.
    #
    # Let me redo Partition 5 carefully.
    # n=2 split into [1,1]: two rows of A
    # m=4 split into [2,2]: two blocks of 2 columns of A, two blocks of 2 rows of B
    # p=7 split into [3,4]: two blocks of 3 and 4 columns of B, two blocks of 3 and 4 of C
    #
    # The sub-problems are:
    #   A[0:1, 0:2] * B[0:2, 0:3] -> C[0:1, 0:3]: 1x2x3, rank 6
    #   A[0:1, 2:4] * B[2:4, 0:3] -> C[0:1, 0:3]: 1x2x3, rank 6
    #   A[0:1, 0:2] * B[0:2, 3:7] -> C[0:1, 3:7]: 1x2x4, rank 8
    #   A[0:1, 2:4] * B[2:4, 3:7] -> C[0:1, 3:7]: 1x2x4, rank 8
    #   A[1:2, 0:2] * B[0:2, 0:3] -> C[1:2, 0:3]: 1x2x3, rank 6
    #   A[1:2, 2:4] * B[2:4, 0:3] -> C[1:2, 0:3]: 1x2x3, rank 6
    #   A[1:2, 0:2] * B[0:2, 3:7] -> C[1:2, 3:7]: 1x2x4, rank 8
    #   A[1:2, 2:4] * B[2:4, 3:7] -> C[1:2, 3:7]: 1x2x4, rank 8
    # Total: 6+6+8+8+6+6+8+8 = 56. That's worse!
    #
    # I see my error. When I split n, m, and p independently, the number of sub-problems
    # is (num_n_blocks) * (num_m_blocks) * (num_p_blocks).
    #
    # For Partition 5: 2 * 2 * 2 = 8 sub-problems.
    #
    # Let me reconsider. The block partition approach:
    # If we split n into n1, n2; m into m1, m2; p into p1, p2, then we have 8 sub-problems:
    #   (n1, m1, p1), (n1, m1, p2), (n1, m2, p1), (n1, m2, p2),
    #   (n2, m1, p1), (n2, m1, p2), (n2, m2, p1), (n2, m2, p2)
    #
    # The total rank is the sum of the ranks of all sub-problems.
    #
    # For n=[1,1], m=[2,2], p=[3,4]:
    #   8 sub-problems, each of size 1x2x3 or 1x2x4.
    #   4 of size 1x2x3 (rank 6 each) and 4 of size 1x2x4 (rank 8 each).
    #   Total: 4*6 + 4*8 = 24 + 32 = 56.
    #
    # This is worse than schoolbook (56) and our current (28).
    #
    # The issue is that splitting all three dimensions multiplies the number of sub-problems.
    # We should only split the inner dimension (m) to get independent sub-problems.
    #
    # If we split m into m1, m2, then we have 2 sub-problems:
    #   (n, m1, p) and (n, m2, p)
    # Total rank: rank(n, m1, p) + rank(n, m2, p)
    #
    # For m=[2,2]: rank(2,2,7) + rank(2,2,7) = 14 + 14 = 28. Same as current.
    #
    # If we also split p into p1, p2, then for each m-block we have 2 sub-problems:
    #   (n, m1, p1), (n, m1, p2), (n, m2, p1), (n, m2, p2)
    # Total rank: sum of all 4.
    #
    # For m=[2,2], p=[3,4]:
    #   rank(2,2,3) + rank(2,2,4) + rank(2,2,3) + rank(2,2,4) = 12 + 16 + 12 + 16 = 56.
    #
    # This is worse.
    #
    # So the best strategy is to only split m, not n or p.
    # And for m=[2,2], we get rank 28.
    #
    # Can we do better by using Strassen on 2x2x2 sub-problems?
    # If we split m=[2,2] and p=[2,2,3], then for each m-block we have 3 sub-problems:
    #   (2,2,2), (2,2,2), (2,2,3)
    # Total: 2 * (7 + 7 + 12) = 2 * 26 = 52.
    #
    # This is worse than 28.
    #
    # I think rank 28 is hard to beat with simple block partitions.
    # Let's just use the current construction and verify it.

    # Use the current Kronecker/schoolbook construction with rank 28
    n, m, p = 2, 4, 7
    r = m * p  # 28
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    col = 0
    for j in range(m):
        for k in range(p):
            r_idx = col
            for i in range(n):
                U[i * m + j, r_idx] = 1.0
            V[j * p + k, r_idx] = 1.0
            for i in range(n):
                W[k * n + i, r_idx] = 1.0
            col += 1

    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(decomposed, T, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p, n*m*p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p, rank):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-END
