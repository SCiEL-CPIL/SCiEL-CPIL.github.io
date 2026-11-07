"""Cochromatic partition solver: bitmask DP + backtracking verification."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80  # safety margin before 90s limit

    # Build adjacency as bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute: for each subset, is it a clique? is it an independent set?
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if rest == 0:
            is_clique[mask] = True
            is_independent[mask] = True
        else:
            if is_clique[rest] and (rest & ~adj[v] & full_mask) == 0:
                is_clique[mask] = True
            non_adj_v = (~adj[v]) & full_mask
            if is_independent[rest] and (rest & non_adj_v) == rest:
                is_independent[mask] = True

    # Precompute valid submasks for each mask, sorted by popcount
    valid_submasks = [[] for _ in range(1 << n)]
    for mask in range(1, 1 << n):
        sub = mask
        while sub > 0:
            if is_clique[sub] or is_independent[sub]:
                valid_submasks[mask].append(sub)
            sub = (sub - 1) & mask
        valid_submasks[mask].sort(key=lambda x: x.bit_count())

    # DP: dp[mask] = minimum number of parts to partition mask
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0

    dp_complete = True
    for mask in range(1, 1 << n):
        best = INF
        for sub in valid_submasks[mask]:
            val = 1 + dp[mask ^ sub]
            if val < best:
                best = val
                if best == 1:
                    break
        dp[mask] = best
        if time.time() > deadline:
            dp_complete = False
            break

    if dp_complete and dp[full_mask] < INF:
        # Reconstruct partition from DP
        result_parts = []
        mask = full_mask
        while mask > 0:
            found = False
            for sub in valid_submasks[mask]:
                if (is_clique[sub] or is_independent[sub]) and dp[mask ^ sub] == dp[mask] - 1:
                    part = []
                    s = sub
                    while s > 0:
                        lsb2 = s & (-s)
                        v2 = lsb2.bit_length() - 1
                        part.append(v2)
                        s ^= lsb2
                    result_parts.append(part)
                    mask ^= sub
                    found = True
                    break
            if not found:
                break

        covered = set()
        for part in result_parts:
            for v in part:
                covered.add(v)
        if covered == set(range(n)):
            return [[int(v) for v in part] for part in result_parts]

    # Fallback: greedy + local search + backtracking verification
    def is_valid_part(part):
        if len(part) <= 1:
            return True
        clique = True
        indep = True
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                u, v = part[i], part[j]
                if (adj[u] >> v) & 1:
                    indep = False
                else:
                    clique = False
                if not clique and not indep:
                    return False
        return True

    def greedy_partition():
        parts = []
        for v in range(n):
            placed = False
            for i, part in enumerate(parts):
                trial = part + [v]
                if is_valid_part(trial):
                    parts[i].append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    def local_search_improve(parts):
        parts = [list(p) for p in parts]
        improved = True
        while improved:
            improved = False
            for i in range(len(parts)):
                for j in range(len(parts)):
                    if i == j:
                        continue
                    for v_idx, v in enumerate(parts[i]):
                        new_j = parts[j] + [v]
                        new_i = parts[i][:v_idx] + parts[i][v_idx + 1:]
                        if is_valid_part(new_j) and (len(new_i) == 0 or is_valid_part(new_i)):
                            parts[i] = new_i
                            parts[j] = new_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            parts = [p for p in parts if len(p) > 0]
        return parts

    def backtrack_k_parts(k):
        """Try to find a valid partition with exactly k parts via backtracking."""
        if time.time() > deadline:
            return None

        vertex_order = sorted(range(n), key=lambda v: -bin(adj[v]).count('1'))
        parts = [0] * k  # bitmasks

        def search(idx):
            if time.time() > deadline:
                return None
            if idx == len(vertex_order):
                result = []
                for p in parts:
                    if p:
                        verts = []
                        temp = p
                        while temp:
                            lsb = temp & (-temp)
                            verts.append(lsb.bit_length() - 1)
                            temp ^= lsb
                        result.append(verts)
                return result

            v = vertex_order[idx]
            vmask = 1 << v

            # Try placing in existing non-empty parts
            for i in range(k):
                if parts[i] != 0:
                    trial = parts[i] | vmask
                    if is_clique[trial] or is_independent[trial]:
                        parts[i] = trial
                        res = search(idx + 1)
                        if res is not None:
                            return res
                        parts[i] = trial ^ vmask

            # Try placing in first empty part
            for i in range(k):
                if parts[i] == 0:
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                    break

            return None

        return search(0)

    parts = greedy_partition()
    parts = local_search_improve(parts)

    # Try to merge pairs of parts
    merged = True
    while merged:
        merged = False
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                merged_part = parts[i] + parts[j]
                if is_valid_part(merged_part):
                    parts[i] = merged_part
                    parts[j] = []
                    merged = True
                    break
            if merged:
                break
        parts = [p for p in parts if len(p) > 0]

    # Try backtracking to verify/improve with fewer parts
    current_k = len(parts)
    for k in range(current_k - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        result = backtrack_k_parts(k)
        if result is not None:
            parts = result
            break

    return [[int(v) for v in part] for part in parts]

# EVOLVE-BLOCK-END
