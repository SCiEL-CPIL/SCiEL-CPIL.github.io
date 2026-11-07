"""
Cochromatic partitions of a pinned random graph.
Uses bitmask DP with memoized branch-and-bound, ordered candidate submasks,
and a greedy upper bound to prune the search.
"""
import time
import sys

sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    deadline = time.time() + 50.0
    n = P['n']
    edges = P['edges']

    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    full_mask = (1 << n) - 1
    size = 1 << n

    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True

    for mask in range(1, size):
        lbit = mask & -mask
        v = lbit.bit_length() - 1
        rest = mask ^ lbit

        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True

        if is_independent[rest] and (adj_mask[v] & rest) == 0:
            is_independent[mask] = True

    valid = [False] * size
    for mask in range(1, size):
        if is_clique[mask] or is_independent[mask]:
            valid[mask] = True

    # Precompute list of valid submasks for each mask, sorted by size descending.
    # For each mask, store list of valid submasks that are subsets of mask.
    # This is memory heavy for large n, but n=14 is fine.
    # Actually, storing all submasks for all masks is 2^n * 2^n which is too much.
    # Instead, we'll generate them on the fly but with better ordering.

    # Greedy upper bound
    def greedy_bound(rem_mask):
        parts = 0
        cur = rem_mask
        while cur:
            lbit = cur & -cur
            # Find largest valid submask containing lbit
            best_sub = lbit
            sub = cur
            while sub:
                if (sub & lbit) == lbit and valid[sub]:
                    if sub.bit_count() > best_sub.bit_count():
                        best_sub = sub
                sub = (sub - 1) & cur
            parts += 1
            cur ^= best_sub
        return parts

    ub = greedy_bound(full_mask)

    memo = {}
    parent_part = {}

    def min_parts(rem_mask):
        if time.time() > deadline:
            return -1

        if rem_mask == 0:
            return 0

        if rem_mask in memo:
            return memo[rem_mask]

        lbit = rem_mask & -rem_mask
        best_val = ub + 1  # Upper bound from greedy

        # Iterate submasks containing lbit, ordered by size descending
        # We collect all valid submasks first, then sort
        candidates = []
        sub = rem_mask
        while sub:
            if (sub & lbit) == lbit and valid[sub]:
                candidates.append(sub)
            sub = (sub - 1) & rem_mask

        # Sort by size descending to find large parts first (better pruning)
        candidates.sort(key=lambda s: s.bit_count(), reverse=True)

        for sub in candidates:
            if time.time() > deadline:
                return -1
            prev_val = min_parts(rem_mask ^ sub)
            if prev_val != -1:
                val = prev_val + 1
                if val < best_val:
                    best_val = val
                    parent_part[rem_mask] = sub
                    if best_val == 1:
                        memo[rem_mask] = 1
                        return 1

        if best_val > ub:
            best_val = ub  # Shouldn't happen, but safety

        memo[rem_mask] = best_val
        return best_val

    result_score = min_parts(full_mask)

    if result_score == -1:
        # Timeout fallback: greedy
        parts = []
        cur = full_mask
        while cur:
            lbit = cur & -cur
            best_sub = lbit
            sub = cur
            while sub:
                if (sub & lbit) == lbit and valid[sub]:
                    if sub.bit_count() > best_sub.bit_count():
                        best_sub = sub
                sub = (sub - 1) & cur
            parts.append(best_sub)
            cur ^= best_sub
        result = []
        for mask in parts:
            verts = [v for v in range(n) if mask & (1 << v)]
            result.append(verts)
        return result

    # Reconstruct
    current = full_mask
    parts_masks = []
    while current > 0:
        p = parent_part.get(current, 0)
        if p == 0:
            p = current & -current
        parts_masks.append(p)
        current ^= p

    result = []
    for mask in parts_masks:
        verts = [v for v in range(n) if mask & (1 << v)]
        result.append(verts)

    return result


if __name__ == "__main__":
    res = construct()
    assert len(res) > 0
    print(f"Found partition with {len(res)} parts: {res}")
