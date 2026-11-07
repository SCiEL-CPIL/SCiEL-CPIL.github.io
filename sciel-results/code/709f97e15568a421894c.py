"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 85
    n = P['n']
    min_verts = P['minimum_vertices']

    # Build full-vertex bitmask and adjacency bitmasks
    full_mask = (1 << n) - 1
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    # Degree within an induced subgraph: popcount of (adj[v] & S)
    pop = int.bit_count

    # Bitset of degree -> bitmask over vertices v (full universe) that have that degree
    # deg_set[d] has bit v set iff deg(v; S) == d for the current S
    deg_set = [0] * n
    for v in range(n):
        d0 = pop(adj[v])  # degree in the full graph = degree in S = full_mask
        # For S = full_mask, we need to initialize per-vertex degrees
    # Initialize for the full set
    cur_deg = [pop(adj[v]) for v in range(n)]
    for v in range(n):
        deg_set[cur_deg[v]] |= 1 << v

    # Track current distinct degrees using a bitmask over degree values
    distinct_mask = sum(1 << cur_deg[v] for v in range(n))
    best_distinct = distinct_mask.bit_count()
    best_mask = full_mask

    t0 = time.time()

    # Iterate subsets in decreasing size order via Gray-code enumeration over all masks
    # We enumerate masks from full_mask down to 0 (decreasing popcount is not guaranteed by mask value order)
    # Instead, we iterate k from n down to min_verts and use combinations for each size,
    # but with incremental degree updates using a smarter approach.

    # Better: use a BFS/DFS over subsets ordered by popcount. Since n=17, total subsets = 2^17 = 131072.
    # We can iterate all masks in order of decreasing popcount using a bucket approach.

    # Group masks by popcount
    by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        pc = mask.bit_count()
        if pc >= min_verts:
            by_size[pc].append(mask)

    # Process from largest to smallest
    for k in range(n, min_verts - 1, -1):
        if time.time() - t0 > 80:
            break
        for mask in by_size[k]:
            if time.time() - t0 > 85:
                break
            # Compute degrees for this mask
            # For each vertex v in mask, deg = popcount(adj[v] & mask)
            # We can track distinct degrees via a bitmask
            dmask = 0
            # Optimization: only check vertices in the mask
            temp = mask
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                d = pop(adj[v] & mask)
                dmask |= 1 << d
                temp ^= low
            cd = dmask.bit_count()
            if cd > best_distinct:
                best_distinct = cd
                best_mask = mask
                # If we've achieved the theoretical max for this size, we can stop early
                # Max distinct degrees for k vertices is at most k, and degrees are in [0, k-1]
                # So max is k. If cd == k, we can't do better for this k.
                # But we still need to check smaller k values? No, since we want to maximize
                # distinct degrees overall, and a larger set can't have fewer than k distinct
                # degrees if it achieves k. Actually distinct degrees <= k always.
                # The maximum possible is k for a k-set. So if best_distinct == k, we've
                # found a perfect set for size k, and no smaller set can beat it (smaller sets
                # have at most k' < k distinct degrees at most... wait, distinct degrees <= k'.
                # So if we have k distinct degrees from a k-set, no smaller set can exceed k.
                # But a larger set is processed first. So once we find best_distinct == k for
                # the largest k, we can potentially stop if no larger k was processed (we process
                # from n down). Actually we process n first, then n-1, etc.
                # The absolute max distinct degrees is at most n. If we find n distinct degrees,
                # we can stop immediately.
                if best_distinct >= n:
                    # Can't do better
                    return [int(i) for i in range(n) if (best_mask >> i) & 1]
                # Also, if best_distinct == k, then for all smaller k' < k, distinct degrees
                # <= k' < k = best_distinct, so we can stop.
                if best_distinct == k:
                    break
        # If best_distinct is already > any possible distinct degrees for smaller k, stop
        # Smaller k can have at most k distinct degrees. So if best_distinct > k-1, no
        # smaller set can beat it.
        if k > min_verts and best_distinct > k - 1:
            break

    # Convert best_mask to list of vertex indices
    result = [int(i) for i in range(n) if (best_mask >> i) & 1]
    if len(result) < min_verts:
        result = list(range(min_verts))
    return result

# EVOLVE-BLOCK-END
