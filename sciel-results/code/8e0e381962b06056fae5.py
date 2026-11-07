"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']

    # Adjacency bitmasks
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1
    size = 1 << n

    # valid[mask]: True iff mask induces a clique OR an independent set.
    # Use recurrence: for mask with LSB v, rest = mask ^ (1<<v)
    # mask is clique iff rest is clique AND v adjacent to all in rest
    # mask is indep iff rest is indep AND v non-adjacent to all in rest
    is_clique_arr = [False] * size
    is_indep_arr = [False] * size
    is_clique_arr[0] = True
    is_indep_arr[0] = True
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        if is_clique_arr[rest] and (adj[v] & rest) == rest:
            is_clique_arr[mask] = True
        if is_indep_arr[rest] and (adj[v] & rest) == 0:
            is_indep_arr[mask] = True
        if not is_clique_arr[mask] and not is_indep_arr[mask]:
            continue
        # valid will be set below
    valid = [is_clique_arr[m] or is_indep_arr[m] for m in range(size)]
    valid[0] = True

    # Subset DP: dp[mask] = minimum number of parts to partition mask.
    INF = 999
    dp = [INF] * size
    dp[0] = 0
    for mask in range(1, size):
        sub = mask
        best = INF
        while sub:
            if valid[sub]:
                prev = dp[mask ^ sub]
                if prev + 1 < best:
                    best = prev + 1
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best
        if time.time() > deadline:
            break

    if dp[full_mask] < INF:
        # Reconstruct a valid partition achieving dp[full_mask].
        parts = []
        mask = full_mask
        while mask:
            sub = mask
            found = False
            while sub:
                if valid[sub] and dp[mask ^ sub] + 1 == dp[mask]:
                    parts.append([v for v in range(n) if (sub >> v) & 1])
                    mask ^= sub
                    found = True
                    break
                sub = (sub - 1) & mask
            if not found:
                # Should not happen; fallback to keep validity.
                parts.append([v for v in range(n) if (mask >> v) & 1])
                break
        # Ensure all parts are Python lists of Python ints
        result = []
        for part in parts:
            result.append([int(v) for v in part])
        return result
    else:
        # Fallback greedy (valid partition) in case DP was cut off.
        es = set()
        for a, b in P['edges']:
            es.add((a, b)); es.add((b, a))
        parts = []
        for v in range(n):
            placed = False
            for i in range(len(parts)):
                part = parts[i]
                trial = part + [v]
                hits = [(u, w) in es for idx, u in enumerate(trial) for w in trial[idx+1:]]
                if all(hits) or not any(hits):
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return [[int(v) for v in part] for part in parts]
# EVOLVE-BLOCK-END
