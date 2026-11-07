"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute: for each vertex, the set of vertices it's adjacent to
    # For a clique: all pairs must be adjacent
    # For an independent set: no pair must be adjacent

    def is_clique(mask):
        """Check if vertices in mask form a clique."""
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if not (adj[verts[i]] & (1 << verts[j])):
                    return False
        return True

    def is_independent(mask):
        """Check if vertices in mask form an independent set."""
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]] & (1 << verts[j]):
                    return False
        return True

    # Precompute all maximal cliques and independent sets
    # Actually, let's just enumerate all subsets and check
    # For n=14, 2^14 = 16384 subsets, very manageable

    full_mask = (1 << n) - 1

    # Precompute validity for all subsets
    # valid_clique[mask] = True if mask forms a clique
    # valid_indep[mask] = True if mask forms an independent set
    valid_clique = [False] * (1 << n)
    valid_indep = [False] * (1 << n)
    valid_clique[0] = True
    valid_indep[0] = True

    for mask in range(1, 1 << n):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb

        # Check clique
        ok_c = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if not (adj[verts[i]] & (1 << verts[j])):
                    ok_c = False
                    break
            if not ok_c:
                break
        valid_clique[mask] = ok_c

        # Check independent set
        ok_i = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]] & (1 << verts[j]):
                    ok_i = False
                    break
            if not ok_i:
                break
        valid_indep[mask] = ok_i

    # Now do BFS/DFS to find minimum number of parts
    # State: remaining mask
    # We want to partition full_mask into minimum number of parts,
    # where each part is either a clique or an independent set.

    # Use BFS with memoization: dp[mask] = min parts to cover mask
    # dp[0] = 0
    # dp[mask] = 1 + min over all valid submasks s of mask: dp[mask ^ s]

    # But we need to be careful: we want the actual partition, not just the count.
    # Let's first find the minimum count, then reconstruct.

    from functools import lru_cache

    # Use iterative deepening or BFS
    # Since n=14, let's use memoized DFS

    memo = {}

    def min_parts(rem_mask):
        """Return minimum number of parts to cover rem_mask, or -1 if impossible."""
        if rem_mask == 0:
            return 0
        if rem_mask in memo:
            return memo[rem_mask]
        if time.time() > deadline:
            memo[rem_mask] = -1  # timeout, treat as impossible
            return -1

        best = n + 1  # infinity
        # Enumerate all submasks of rem_mask
        sub = rem_mask
        while sub:
            if valid_clique[sub] or valid_indep[sub]:
                rest = rem_mask ^ sub
                r = min_parts(rest)
                if r >= 0:
                    if r + 1 < best:
                        best = r + 1
                        if best == 1:  # can't do better than 1
                            break
            sub = (sub - 1) & rem_mask

        memo[rem_mask] = best
        return best

    # Try to find the minimum
    result_count = min_parts(full_mask)

    # If timeout or failed, fall back to greedy
    if result_count < 0 or result_count > n:
        # Greedy fallback
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in [(e[0], e[1]) for e in P['edges']] | {(e[1], e[0]) for e in P['edges']}
                        for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return [list(p) for p in parts]

    # Reconstruct the partition
    # We need to find the actual parts, not just the count
    # Re-do the search but track the choices

    memo2 = {}

    def find_partition(rem_mask, target_count):
        """Find a partition of rem_mask into exactly target_count parts."""
        if rem_mask == 0:
            return [] if target_count == 0 else None
        if target_count == 0 and rem_mask != 0:
            return None
        if (rem_mask, target_count) in memo2:
            return memo2[(rem_mask, target_count)]
        if time.time() > deadline:
            memo2[(rem_mask, target_count)] = None
            return None

        sub = rem_mask
        while sub:
            if valid_clique[sub] or valid_indep[sub]:
                rest = rem_mask ^ sub
                rest_parts = find_partition(rest, target_count - 1)
                if rest_parts is not None:
                    result = [list(range(n))]  # placeholder
                    # Actually, let's just return the list of vertex lists
                    verts_in_sub = []
                    m = sub
                    while m:
                        lb = m & (-m)
                        v = lb.bit_length() - 1
                        verts_in_sub.append(v)
                        m ^= lb
                    memo2[(rem_mask, target_count)] = [verts_in_sub] + rest_parts
                    return [verts_in_sub] + rest_parts
            sub = (sub - 1) & rem_mask

        memo2[(rem_mask, target_count)] = None
        return None

    partition = find_partition(full_mask, result_count)

    if partition is None:
        # Fallback to greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                es = set()
                for e in P['edges']:
                    es.add((e[0], e[1]))
                    es.add((e[1], e[0]))
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return [list(p) for p in parts]

    # Convert to list of lists of Python ints
    return [list(part) for part in partition]
# EVOLVE-BLOCK-END
