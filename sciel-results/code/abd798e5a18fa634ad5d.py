"""Optimized cochromatic partition using DP over valid subsets with deadline guard."""

import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 75.0

    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    size = 1 << n

    # Precompute which subsets are cliques and which are independent sets
    is_clique = [False] * size
    is_indep = [False] * size

    is_clique[0] = True
    is_indep[0] = True
    for i in range(n):
        mask = 1 << i
        is_clique[mask] = True
        is_indep[mask] = True

    for mask in range(1, size):
        if mask & (mask - 1) == 0:
            continue
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True

        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True

    # Combine into single validity array for faster inner-loop access
    is_valid = [is_clique[m] or is_indep[m] for m in range(size)]

    if time.time() > deadline:
        return [[i] for i in range(n)]

    # Collect valid subsets for potential use
    valid_subsets = [m for m in range(1, size) if is_valid[m]]

    # DP for minimum number of parts
    INF = n + 1
    dp = [INF] * size
    choice = [0] * size
    dp[0] = 0

    # Local references for speed
    _is_valid = is_valid
    _dp = dp
    _choice = choice
    _n = n

    for mask in range(1, size):
        # If mask itself is a valid part, we're done
        if _is_valid[mask]:
            _dp[mask] = 1
            _choice[mask] = mask
            continue

        sub = mask
        best_val = INF
        best_sub = 0
        while sub:
            if _is_valid[sub]:
                rest = mask ^ sub
                val = _dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:
                        break
            sub = (sub - 1) & mask
        _dp[mask] = best_val
        _choice[mask] = best_sub

        if (mask & 255) == 0 and time.time() > deadline:
            break

    full_mask = size - 1
    if dp[full_mask] >= INF:
        return [[i] for i in range(n)]

    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        part_mask = choice[current_mask]
        if part_mask == 0:
            return [[i] for i in range(n)]
        part_vertices = []
        for i in range(n):
            if (part_mask >> i) & 1:
                part_vertices.append(i)
        parts.append(part_vertices)
        current_mask ^= part_mask

    result = [list(p) for p in parts]
    for part in result:
        for i in range(len(part)):
            part[i] = int(part[i])

    return result
