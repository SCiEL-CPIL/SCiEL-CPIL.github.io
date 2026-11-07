"""Rewritten program: exact cochromatic partition via bitmask DP with deadline guard."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build():
    n = P['n']
    edges = P['edges']
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    size = 1 << n
    # valid[mask] = True if the vertex set is a clique or an independent set
    valid = [False] * size
    valid[0] = True
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        if rest == 0:
            valid[mask] = True
            continue
        # check clique: v connected to all in rest
        if (adj[v] & rest) == rest:
            valid[mask] = True
        else:
            # check independent: v not connected to any in rest
            if (adj[v] & rest) == 0:
                valid[mask] = True
    return n, adj, valid, size

def construct():
    t0 = time.time()
    DEADLINE = t0 + 75.0  # safety margin under 90s kill
    n, adj, valid, size = _build()
    full = size - 1

    # Fallback greedy partition (valid, always available)
    def greedy():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                ok = True
                for u in part:
                    if ((u, v) in set()) :
                        pass
                # recompute: check clique or independent with v
                hits = 0
                total = 0
                for u in part:
                    total += 1
                    if adj[u] & (1 << v):
                        hits += 1
                if hits == 0 or hits == total:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    # DP over subsets: dp[mask] = min number of valid parts to cover mask
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    # Precompute list of valid masks sorted by popcount (smaller first helps pruning)
    # We'll iterate masks in increasing order; for each, extend by a valid subset of complement
    # To keep it fast, we iterate valid submasks of the remaining set.
    # Precompute valid masks as a list for iteration
    valid_masks = [m for m in range(1, size) if valid[m]]

    # Better: for each mask, dp[mask] = min over valid submask s of mask: dp[mask ^ s] + 1
    # Iterate mask from 1..full, and for each valid submask s of mask
    for mask in range(1, size):
        if time.time() > DEADLINE:
            break
        best = INF
        # enumerate valid submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best

    if dp[full] >= INF:
        # Should not happen (singletons are valid), but fallback
        return greedy()

    # Reconstruct partition from dp
    parts = []
    mask = full
    while mask:
        sub = mask
        while sub:
            if valid[sub] and dp[mask ^ sub] + 1 == dp[mask]:
                parts.append([i for i in range(n) if (mask & (1 << i)) and (sub & (1 << i))])
                mask ^= sub
                break
            sub = (sub - 1) & mask
        if not parts and mask:
            # safety
            break

    # Ensure every vertex appears exactly once
    seen = 0
    for part in parts:
        for v in part:
            seen |= (1 << v)
    if seen != full:
        return greedy()

    # Convert to pure Python ints / lists (already are)
    return parts
# EVOLVE-BLOCK-END
