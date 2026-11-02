"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    schoolbook_rank = n * m * p  # 72

    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W, r):
        """Check if U, V, W form a valid rank-r decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_decomposition_fast(U, V, W, r):
        """Faster check using numpy."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: partition 3x4x6 into three disjoint 3x2x2 sub-tensors.
    # m=4 -> two blocks of size 2 (j=0,1) and (j=2,3)
    # p=6 -> three blocks of size 2 (k=0,1), (k=2,3), (k=4,5)
    # We need to pair up the j-blocks and k-blocks so that each (i, j_block, k_block)
    # triple is covered exactly once.
    #
    # Since m has 2 j-blocks and p has 3 k-blocks, we can't do a clean 3-way split
    # into 3x2x2 blocks. Instead, let's think differently:
    #
    # Alternative: split p=6 into three groups of 2. For each group of k indices,
    # we need to cover all (i, j) pairs. That's a 3x4x2 tensor for each k-group.
    # A 3x4x2 tensor has "schoolbook" rank 24. But we can do better.
    #
    # Actually, let me reconsider. The directive says to use rank-18 for 3x2x2.
    # Let's try: split m=4 into two halves (j in {0,1} and j in {2,3}),
    # and p=6 into three halves (k in {0,1}, {2,3}, {4,5}).
    #
    # For the full tensor, we need: for each (i,j,k), sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    #
    # If we partition the columns into groups, one per (j_block, k_block) pair,
    # we get 2*3 = 6 groups. But that gives rank 6 * rank(3x2x2) which is too much.
    #
    # Better approach: The tensor T_{i,j,k} = 1 for all i,j,k.
    # We can write T = sum over blocks. If we split j into {0,1} and {2,3},
    # and k into {0,1}, {2,3}, {4,5}, then T is the sum of 6 sub-tensors,
    # each being 3x2x2 (all ones). If each 3x2x2 all-ones tensor has rank R,
    # total rank is 6R.
    #
    # But wait - we can do better by overlapping. Let me just try a direct
    # algebraic construction for the full 3x4x6 tensor.
    #
    # Key insight: T_{i,j,k} = 1 = 1 * 1 * 1 for all (i,j,k).
    # We can try to find a low-rank decomposition directly.
    #
    # Let's use the fact that for the all-ones tensor of shape (n,m,p),
    # we can write it as a sum of rank-1 terms where each term is
    # u_i * v_j * w_k.
    #
    # A simple approach: use the identity 1 = sum of products.
    # For example, with 3x4x6:
    # We can try to find vectors u (length 12), v (length 24), w (length 18)
    # such that sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c] = 1.
    #
    # Let's try a structured approach using Kronecker products or similar.
    #
    # Actually, let me just implement the block decomposition as directed:
    # Split into 3 blocks of (3x4x2) each (by k-index), and for each 3x4x2 block,
    # further split into two (3x2x2) blocks (by j-index).
    # Total: 6 blocks of 3x2x2, each needing some rank.
    #
    # For a 3x2x2 all-ones tensor, what's the minimal rank?
    # The tensor has 12 entries, all equal to 1.
    # Lower bound: max(3*2, 2*2, 2*3) = max(6,4,6) = 6.
    # Can we do rank 6? Let's try.
    #
    # Actually, for the all-ones tensor of shape (a,b,c), the rank is
    # at most a*b*c (schoolbook) but can be lower.
    # For 3x2x2: let's try to find a rank-6 decomposition.
    #
    # T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # Try rank 6: one term per (i, j, k) combination? That's schoolbook = 12.
    # We need to do better.
    #
    # Let me try a different approach. Since the tensor is all ones,
    # we can use the fact that 1 = (1/3) * sum_i' delta_{i,i'} for any i.
    # This suggests using indicator functions.
    #
    # Actually, let me just try to find any valid decomposition with rank < 72
    # using a more systematic search, and fall back to schoolbook.

    # First, try the block-based construction
    # Split j into two groups of 2, k into three groups of 2
    # For each (j_group, k_group) pair, we have a 3x2x2 all-ones sub-tensor
    # We'll use a rank-8 decomposition for each 3x2x2 block (to be safe)
    # Total rank: 6 * 8 = 48

    def make_3x2x2_decomposition(rank_per_block=8):
        """Create a rank-r decomposition of the 3x2x2 all-ones tensor.
        Returns (U_block, V_block, W_block) with shapes (6, r), (4, r), (6, r)
        where U is indexed by (i,j) with i in 0..2, j in 0..1
        V is indexed by (j,k) with j in 0..1, k in 0..1
        W is indexed by (k,i) with k in 0..1, i in 0..2
        """
        # Schoolbook for 3x2x2: rank 12
        # Try to do better with rank 8 or less
        # For now, use schoolbook (rank 12) as a baseline
        r = rank_per_block
        U_b = [[0.0] * r for _ in range(6)]  # i*2+j
        V_b = [[0.0] * r for _ in range(4)]  # j*2+k
        W_b = [[0.0] * r for _ in range(6)]  # k*3+i

        # Schoolbook: one term per (i,j,k)
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    if col < r:
                        U_b[i * 2 + j][col] = 1.0
                        V_b[j * 2 + k][col] = 1.0
                        W_b[k * 3 + i][col] = 1.0
                        col += 1
        return U_b, V_b, W_b

    # Build the full decomposition using blocks
    # j_groups: [0,1] and [2,3]
    # k_groups: [0,1], [2,3], [4,5]
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5)]

    block_rank = 12  # schoolbook for 3x2x2
    total_rank = len(j_groups) * len(k_groups) * block_rank  # 6 * 12 = 72

    # That's the same as schoolbook. Let's try to reduce block_rank.
    # For 3x2x2, can we do rank < 12?
    # Let's try rank 8 for each block.

    # Actually, let's just do a direct search for a good decomposition
    # of the full 3x4x6 tensor, using the block structure as initialization.

    # Initialize with schoolbook
    best_result = make_schoolbook()
    best_rank = schoolbook_rank

    # Try to improve using local search / random restarts
    # Focus on finding rank < 72

    # Strategy: Start from schoolbook, try to merge columns
    # Two schoolbook columns (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if we can find new U,V,W values that reproduce both contributions.

    # For now, let's try a random search with structured initialization
    random.seed(42)

    half_ints = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    # Try ranks from 48 down to 72 (we know 72 works)
    for rank in range(48, schoolbook_rank + 1):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        max_attempts = int(remaining * 2)
        max_attempts = max(max_attempts, 5)

        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Initialize with block structure
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]

            # Place schoolbook terms in the first min(rank, 72) columns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U[i * m + j][col] = 1.0
                            V[j * p + k][col] = 1.0
                            W[k * n + i][col] = 1.0
                            col += 1

            # If rank < 72, we need to combine some terms
            # Add random perturbations to non-schoolbook columns
            if rank < 72:
                for c in range(rank):
                    if random.random() < 0.3:
                        U[random.randint(0, n * m - 1)][c] += random.choice(half_ints)
                        V[random.randint(0, m * p - 1)][c] += random.choice(half_ints)
                        W[random.randint(0, p * n - 1)][c] += random.choice(half_ints)

            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break

        if found:
            continue

    # Final fallback: schoolbook
    if best_rank >= schoolbook_rank:
        best_result = make_schoolbook()
        best_rank = schoolbook_rank

    return best_result
# EVOLVE-BLOCK-END
