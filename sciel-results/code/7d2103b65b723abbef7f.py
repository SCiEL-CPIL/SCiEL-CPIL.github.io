"""Rewritten program: exact DP for cochromatic number (min parts that are cliques or independent sets)."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    DEADLINE = start_time + 70  # safety margin before 90s kill

    n = P['n']
    full_mask = (1 << n) - 1

    # Precompute adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute complement adjacency (for independent sets)
    # comp_adj[v] has bits for vertices NOT adjacent to v (and v != itself)
    comp_adj = [0] * n
    for v in range(n):
        # All vertices except v
        all_except_v = full_mask ^ (1 << v)
        comp_adj[v] = all_except_v & ~adj[v]

    # A subset S is a clique iff for every v in S, S \ {v} is a subset of adj[v]
    # A subset S is independent iff for every v in S, S \ {v} is a subset of comp_adj[v]
    # Precompute is_clique[S] and is_indep[S] for all S
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True

    for S in range(1, 1 << n):
        # Pick lowest set bit
        v = (S & -S).bit_length() - 1
        rest = S ^ (1 << v)
        is_clique[S] = is_clique[rest] and ((rest & ~adj[v]) == 0)
        is_indep[S] = is_indep[rest] and ((rest & ~comp_adj[v]) == 0)

    # is_good[S] = is_clique[S] or is_indep[S]
    is_good = [False] * (1 << n)
    for S in range(1 << n):
        is_good[S] = is_clique[S] or is_indep[S]

    # DP: dp[S] = minimum number of parts to cover subset S
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0

    # For each subset S, try all submasks T of S that are "good" (clique or independent)
    # dp[S] = min over good submasks T of S of (dp[S ^ T] + 1)
    # This is O(3^n) which for n=14 is about 4.78 million, very fast.

    for S in range(1, 1 << n):
        # Iterate over all submasks of S
        T = S
        best = INF
        while T:
            if is_good[T]:
                val = dp[S ^ T] + 1
                if val < best:
                    best = val
            T = (T - 1) & S
        dp[S] = best

    # Now reconstruct the partition
    # dp[full_mask] gives the minimum number of parts
    min_parts = dp[full_mask]

    # Reconstruct: greedily pick a good submask T of remaining such that dp[remaining] = dp[remaining^T] + 1
    partition = []
    remaining = full_mask
    while remaining:
        if time.time() > DEADLINE:
            # Fallback: should not happen for n=14
            break
        T = remaining
        found = False
        while T:
            if is_good[T] and dp[remaining ^ T] + 1 == dp[remaining]:
                partition.append(T)
                remaining ^= T
                found = True
                break
            T = (T - 1) & remaining
        if not found:
            # Should not happen; fallback to singleton
            # Pick lowest vertex
            v = (remaining & -remaining).bit_length() - 1
            partition.append(1 << v)
            remaining ^= (1 << v)

    # Convert bitmasks to lists of vertices
    result = []
    for mask in partition:
        verts = []
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m ^= (1 << v)
        result.append(verts)

    # Ensure types are correct: list of lists of Python ints
    return result

# EVOLVE-BLOCK-END
