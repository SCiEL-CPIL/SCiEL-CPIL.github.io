"""Cochromatic partition solver with optimized bitmask DP."""
import time

DEADLINE = 85.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute for every subset whether it is a clique and/or independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    # Single vertices are both cliques and independent sets
    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_indep[m] = True

    # Process all masks incrementally
    for mask in range(1, 1 << n):
        if mask.bit_count() == 1:
            continue
        if is_clique[mask] or is_indep[mask]:
            continue

        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest]:
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
                continue

        if is_indep[rest]:
            if (rest & adj[v]) == 0:
                is_indep[mask] = True
                continue

    # Precompute valid_part and also precompute valid submasks for each mask
    valid_part = [False] * (1 << n)
    for mask in range(1, 1 << n):
        valid_part[mask] = is_clique[mask] or is_indep[mask]

    # DP to find minimum number of parts
    min_parts = [n + 1] * (1 << n)
    choice = [0] * (1 << n)
    min_parts[0] = 0

    # Track time more frequently
    time_check_interval = 1000  # check every 1000 masks
    for mask in range(1, 1 << n):
        if (mask & (time_check_interval - 1)) == 0 and time.time() - start_time > DEADLINE - 5.0:
            break

        lsb = mask & (-mask)
        best = n + 1
        best_sub = 0

        # Enumerate submasks containing lsb
        # Trick: sub = (sub - 1) & mask, but only consider those with lsb set
        # We can do: for each submask of (mask ^ lsb), the valid sub is sub | lsb
        rest_mask = mask ^ lsb
        sub = rest_mask
        while True:
            candidate = sub | lsb
            if valid_part[candidate]:
                rem = mask ^ candidate
                val = min_parts[rem] + 1
                if val < best:
                    best = val
                    best_sub = candidate
                    if best == 1:
                        break
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask

        min_parts[mask] = best
        choice[mask] = best_sub

    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            # Fallback: greedily find a valid part
            # Try to find a clique or independent set containing the lowest vertex
            v = (mask & (-mask)).bit_length() - 1
            # Try clique first
            clique_mask = 1 << v
            for u in range(v + 1, n):
                if (mask >> u) & 1 and (adj[v] >> u) & 1:
                    clique_mask |= (1 << u)
            # Check if it's actually a clique
            is_valid = True
            temp = clique_mask
            while temp:
                b = temp & (-temp)
                vv = b.bit_length() - 1
                rest_t = temp ^ b
                if rest_t & ~adj[vv]:
                    is_valid = False
                    break
                temp ^= b
            if is_valid:
                sub = clique_mask
            else:
                # Try independent set
                indep_mask = 1 << v
                for u in range(v + 1, n):
                    if (mask >> u) & 1 and not ((adj[v] >> u) & 1):
                        indep_mask |= (1 << u)
                # Verify
                is_valid = True
                temp = indep_mask
                while temp:
                    b = temp & (-temp)
                    vv = b.bit_length() - 1
                    rest_t = temp ^ b
                    if rest_t & adj[vv]:
                        is_valid = False
                        break
                    temp ^= b
                if is_valid:
                    sub = indep_mask
                else:
                    # Just take vertex v
                    sub = 1 << v
        verts = []
        temp = sub
        while temp:
            lsb_t = temp & (-temp)
            verts.append(lsb_t.bit_length() - 1)
            temp ^= lsb_t
        parts.append(verts)
        mask ^= sub

    # Ensure types are correct: Python int, list
    final_result = []
    for part in parts:
        final_result.append([int(v) for v in part])

    return final_result

# EVOLVE-BLOCK-END
