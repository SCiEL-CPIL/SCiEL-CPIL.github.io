"""Rewritten program: stronger algorithm for the 101-residue covering task."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10

    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Enumerate all non-empty subsets: map residue -> list of (size, mask)
    res_to_subsets = defaultdict(list)
    for mask in range(1, 1 << N):
        if _elapsed() > _DEADLINE:
            break
        s = 0
        m = mask
        size = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
        r = s % MOD
        res_to_subsets[r].append((size, mask))

    # For each residue, sort subsets by size
    for r in res_to_subsets:
        res_to_subsets[r].sort()

    # Check if a residue has at least one subset of size <= max_size
    def residue_has_subset(r, max_size):
        if r not in res_to_subsets:
            return False
        for sz, mask in res_to_subsets[r]:
            if sz <= max_size:
                return True
            else:
                break
        return False

    # Hopcroft-Karp algorithm for maximum bipartite matching
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] contains right nodes adjacent to left node i
        n_left: number of left nodes
        n_right: number of right nodes
        Returns matching: match_l[i] = right node matched to left node i, or -1
        """
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            from collections import deque
            q = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == INF:
                        dist[pair_v[v]] = dist[u] + 1
                        q.append(pair_v[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u

    # Try to find a matching for a given max_size using Hopcroft-Karp
    def try_match(max_size):
        # Build adjacency: left = residues (0..100), right = masks
        # But we need to handle the fact that not all residues may have subsets
        adj = [None] * MOD
        for r in range(MOD):
            if not residue_has_subset(r, max_size):
                return None
            adj[r] = [mask for (sz, mask) in res_to_subsets[r] if sz <= max_size]
            if not adj[r]:
                return None

        # Right side is masks 0..(1<<N)-1, but we only care about masks that appear
        # For Hopcroft-Karp, we need consistent indexing
        # Let's use the mask value as the right node index
        n_right = 1 << N
        match_l = hopcroft_karp(adj, MOD, n_right)

        # Check if all residues are matched
        for r in range(MOD):
            if match_l[r] == -1:
                return None
        return match_l

    # Binary search on max_size
    lo, hi = 1, N
    best_K = N
    while lo <= hi:
        if _elapsed() > _DEADLINE:
            break
        mid = (lo + hi) // 2
        if try_match(mid) is not None:
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Now construct the actual assignment using max_size = best_K
    match_l = try_match(best_K)
    if match_l is None:
        # Fallback: use the smallest subset for each residue
        match_l = [None] * MOD
        for r in range(MOD):
            if res_to_subsets[r]:
                match_l[r] = res_to_subsets[r][0][1]
            else:
                match_l[r] = (1 << N) - 1

    # Build rows
    rows = [None] * MOD
    for r in range(MOD):
        assigned_mask = match_l[r]
        terms = [x for x in range(1, N + 1) if (assigned_mask >> (x - 1)) & 1]
        rows[r] = terms

    return rows

# EVOLVE-BLOCK-END
