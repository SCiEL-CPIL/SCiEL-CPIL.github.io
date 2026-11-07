import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 55.0  # safety margin under 60s wall-clock
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    total_masks = 1 << n
    is_clique = [False] * total_masks
    is_indep = [False] * total_masks
    valid = [False] * total_masks

    is_clique[0] = True
    is_indep[0] = True
    valid[0] = True

    # Incremental validity: O(n * 2^n)
    for m in range(1, total_masks):
        lsb = m & -m
        v = lsb.bit_length() - 1
        rest = m ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[m] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[m] = True
        if is_clique[m] or is_indep[m]:
            valid[m] = True

    # Fallback greedy (only used if DP times out)
    def greedy():
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(a, b) in es for i, a in enumerate(trial) for b in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    # Exact minimum set cover DP: dp[mask] = min #valid parts to cover mask
    INF = 10 ** 9
    dp = [INF] * total_masks
    choice = [0] * total_masks
    dp[0] = 0

    for mask in range(1, total_masks):
        best = INF
        best_sub = 0
        sub = mask
        while sub:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_sub = sub
                    if best == 1:
                        break
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        dp[mask] = best
        choice[mask] = best_sub
        if time.time() > deadline:
            break

    # Reconstruct if we completed the DP
    if time.time() <= deadline and dp[(1 << n) - 1] < INF:
        cur = (1 << n) - 1
        partition_masks = []
        while cur:
            s = choice[cur]
            if s == 0:
                break
            partition_masks.append(s)
            cur ^= s
        result = []
        for mask in partition_masks:
            verts = [int(v) for v in range(n) if mask & (1 << v)]
            if verts:
                result.append(verts)
        # sanity: every vertex exactly once
        flat = [v for part in result for v in part]
        if sorted(flat) == list(range(n)) and len(result) > 0:
            return result

    return greedy()
# EVOLVE-BLOCK-END
