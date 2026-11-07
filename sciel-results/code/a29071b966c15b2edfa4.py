"""Improved cochromatic partition search with exact DP lower bound and memoized DFS."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute: for each vertex, its non-neighbors (for independent set extension)
    non_adj = [full_mask ^ (1 << v) ^ adj[v] for v in range(n)]

    # is_clique[mask]: True if vertices in mask form a clique
    # is_indep[mask]: True if vertices in mask form an independent set
    # We'll compute these on-the-fly with caching for subsets we encounter.

    # DP for chromatic number of G (using maximal independent sets)
    # dp[mask] = min number of independent sets to cover mask
    # We use the standard subset DP: dp[0]=0, dp[mask] = 1 + min{dp[mask \ S] : S is a maximal independent subset of mask}
    # But that's expensive. Instead use: dp[mask] = 1 + min{dp[mask \ S] : S is an independent set contained in mask, S is maximal in mask}
    # Simpler: dp[mask] = min over all independent subsets S of mask: 1 + dp[mask ^ S]
    # For n=14, 2^14 = 16384 subsets, and for each we iterate submasks. That's ~3^14 ≈ 4.7M, fine.

    # Precompute is_independent for all masks
    is_indep = [False] * (1 << n)
    is_indep[0] = True
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        is_indep[mask] = is_indep[rest] and (adj[v] & rest) == 0

    # Precompute is_clique for all masks
    is_clique = [False] * (1 << n)
    is_clique[0] = True
    for mask in range(1, 1 << n):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        # v must be adjacent to all in rest
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)

    # Chromatic number of G via DP over independent sets
    # dp[mask] = min parts (independent sets) to cover mask
    dp_chi = [0] * (1 << n)
    for mask in range(1, 1 << n):
        best = n  # upper bound
        sub = mask
        while sub:
            if is_indep[sub]:
                val = 1 + dp_chi[mask ^ sub]
                if val < best:
                    best = val
            sub = (sub - 1) & mask
        dp_chi[mask] = best
    chi_G = dp_chi[full_mask]

    # Chromatic number of complement graph (clique cover of G)
    # dp2[mask] = min number of cliques to cover mask
    dp_chi2 = [0] * (1 << n)
    for mask in range(1, 1 << n):
        best = n
        sub = mask
        while sub:
            if is_clique[sub]:
                val = 1 + dp_chi2[mask ^ sub]
                if val < best:
                    best = val
            sub = (sub - 1) & mask
        dp_chi2[mask] = best
    chi_complement = dp_chi2[full_mask]

    lower_bound = max(chi_G, chi_complement)

    # Now search for a valid partition with k parts, starting from lower_bound
    # Each part must be a clique OR an independent set.
    # DFS: assign vertices to parts. State: (unassigned_mask, parts_remaining)
    # Memo: set of (unassigned_mask, parts_remaining) that are infeasible.

    memo_fail = set()

    def dfs(unassigned, parts_left):
        """Try to cover 'unassigned' vertices with 'parts_left' parts (each a clique or independent set)."""
        if unassigned == 0:
            return parts_left >= 0
        if parts_left == 0:
            return False
        key = (unassigned, parts_left)
        if key in memo_fail:
            return False

        # Find all maximal cliques and independent sets within unassigned
        # We want to pick one part (a clique or independent set) from unassigned,
        # then recurse. To be efficient, we enumerate candidate subsets.

        # Generate candidate parts: all maximal independent sets and maximal cliques in unassigned
        # For simplicity, enumerate all independent subsets and clique subsets,
        # but only consider "maximal" ones to reduce branching.
        # Actually, we can just try all valid subsets (clique or indep) and pick the best.
        # For n=14, the number of subsets is manageable.

        # Collect all valid candidate parts (clique or independent set) that are subsets of unassigned
        # To reduce branching, only consider maximal ones.
        candidates = []
        sub = unassigned
        while sub:
            if is_indep[sub] or is_clique[sub]:
                # Check if sub is maximal (in the sense that no vertex in unassigned\sub can be added)
                # For independent set: can we add any vertex from unassigned\sub that is non-adjacent to all in sub?
                # For clique: can we add any vertex from unassigned\sub that is adjacent to all in sub?
                # We'll just collect all and let the search handle it, but prefer larger ones.
                candidates.append(sub)
            sub = (sub - 1) & unassigned

        # Sort candidates by size descending to try larger parts first
        candidates.sort(key=lambda s: -s.bit_count())

        # Deduplicate: we don't need to try the same subset twice
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            # Try this candidate as one part
            if dfs(unassigned ^ cand, parts_left - 1):
                return True

        memo_fail.add(key)
        return False

    best_k = n  # fallback
    for k in range(lower_bound, n + 1):
        memo_fail.clear()
        if dfs(full_mask, k):
            best_k = k
            break

    # Now we need to actually construct the partition.
    # Re-run DFS but record the choices.
    memo_fail.clear()
    choices = []

    def dfs_record(unassigned, parts_left):
        """Same as dfs but records the partition."""
        if unassigned == 0:
            return parts_left >= 0
        if parts_left == 0:
            return False
        key = (unassigned, parts_left)
        if key in memo_fail:
            return False

        sub = unassigned
        candidates = []
        while sub:
            if is_indep[sub] or is_clique[sub]:
                candidates.append(sub)
            sub = (sub - 1) & unassigned
        candidates.sort(key=lambda s: -s.bit_count())

        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if dfs_record(unassigned ^ cand, parts_left - 1):
                choices.append(cand)
                return True

        memo_fail.add(key)
        return False

    # Reset and record
    choices.clear()
    memo_fail.clear()
    dfs_record(full_mask, best_k)

    # Convert masks to lists of vertices
    parts = []
    for mask in reversed(choices):
        verts = [v for v in range(n) if (mask >> v) & 1]
        parts.append(verts)

    # Ensure all parts are nonempty and valid
    # They should be by construction.
    return parts
