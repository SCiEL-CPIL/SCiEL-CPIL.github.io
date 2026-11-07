"""Bitmask-based exact search for minimum cochromatic partition (pinned graph)."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    edges = P['edges']

    # Build adjacency as list of int bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    full = (1 << n) - 1

    # Precompute for each vertex the "clique-compatible" and "indep-compatible" masks
    # A vertex u can join a clique containing only vertices from adj[u]
    # A vertex u can join an indep set containing only vertices from (~adj[u]) & full

    # Enumerate all non-empty subsets and classify
    # For n=14, 2^14 - 1 = 16383 subsets, each check is O(k) where k = popcount
    # Total work ~ sum over all subsets of popcount(subset) = n * 2^(n-1) = 14 * 8192 = 114688
    # This is very fast.

    clique_set = set()
    indep_set = set()

    for mask in range(1, full + 1):
        # Check clique
        ok = True
        m = mask
        while m and ok:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m ^= lsb
            if m & ~adj[u]:
                ok = False
        if ok:
            clique_set.add(mask)

        # Check independent set
        ok = True
        m = mask
        while m and ok:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m ^= lsb
            if m & adj[u]:
                ok = False
        if ok:
            indep_set.add(mask)

    valid_set = clique_set | indep_set

    # DP over masks: dp[mask] = min parts to partition mask
    # Use iterative DP in increasing order of popcount, or BFS/DFS with memo.
    # For n=14, we can do bottom-up by iterating masks from 0 to full,
    # but the transition is "for each valid submask s of mask", which is expensive.
    # Better: use the fact that we can pick ANY valid subset of mask as one part.
    # dp[mask] = 1 + min_{s in valid, s subset mask, s != 0} dp[mask ^ s]
    # But iterating all valid submasks for each mask is expensive.

    # Alternative: iterative deepening DFS with memoization and pruning.
    # Or: since n is small, we can enumerate all valid submasks for each mask.

    # Let's use a different approach: for each mask, iterate over all valid masks that are subsets.
    # Precompute for each valid mask v, the list of masks that contain v? No, too much.

    # Better approach: iterate mask from 0 to full, for each valid submask s of mask (s != 0),
    # update dp[mask] = min(dp[mask], 1 + dp[mask ^ s]).
    # To iterate submasks of mask: standard submask enumeration.

    import sys
    sys.setrecursionlimit(100000)

    INF = float('inf')
    dp = [INF] * (full + 1)
    dp[0] = 0
    choice = [0] * (full + 1)  # choice[mask] = the valid submask chosen as one part

    # Iterate masks in increasing order of popcount (or just 1..full, since mask^s < mask always? No, not necessarily by value, but by popcount yes)
    # Actually, mask ^ s has fewer bits than mask, so if we iterate by popcount, we're fine.
    # Let's iterate by popcount.
    from collections import defaultdict
    masks_by_popcount = defaultdict(list)
    for mask in range(1, full + 1):
        masks_by_popcount[mask.bit_count()].append(mask)

    max_pop = n
    for pop in range(1, max_pop + 1):
        for mask in masks_by_popcount[pop]:
            best = INF
            best_s = 0
            # Enumerate all non-empty submasks of mask
            sub = mask
            while sub:
                if sub in valid_set:
                    rest = mask ^ sub
                    val = 1 + dp[rest]
                    if val < best:
                        best = val
                        best_s = sub
                sub = (sub - 1) & mask
            dp[mask] = best
            choice[mask] = best_s

    # Reconstruct
    result = []
    cur = full
    while cur:
        s = choice[cur]
        vertices = [i for i in range(n) if (s >> i) & 1]
        result.append(vertices)
        cur ^= s

    return result

# EVOLVE-BLOCK-END
