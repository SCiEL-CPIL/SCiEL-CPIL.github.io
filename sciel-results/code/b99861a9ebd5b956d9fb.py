# EVOLVE-BLOCK-START
import time as _time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = _time.monotonic() + 85.0


def _time_left():
    return _time.monotonic() < _DEADLINE


def _bipartite_max_matching(adj_left, n_right):
    """Kuhn's algorithm. adj_left: list of lists of right indices.
    Returns matching_left (list of right or -1) if a matching saturating
    every left node exists, else None."""
    n_left = len(adj_left)
    if n_left == 0:
        return [None] * 0
    match_r = [-1] * n_right
    match_l = [-1] * n_left

    def try_kuhn(u, seen):
        for v in adj_left[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or try_kuhn(match_r[v], seen):
                match_r[v] = u
                match_l[u] = v
                return True
        return False

    # Try to match all left nodes
    for u in range(n_left):
        seen = [False] * n_right
        if not try_kuhn(u, seen):
            return None
    return match_l


def _max_clique_bruteforce(adj, n):
    """Return the largest clique as a sorted list (n is small)."""
    best = []
    verts = list(range(n))
    # Bron–Kerbosch with pivot
    def bronk(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        # pivot
        if P or X:
            pivot_set = P | X
            # choose pivot with max degree in P
            pivot = max(pivot_set, key=lambda x: len(P & adj[x]))
        else:
            return
        candidates = P - adj[pivot]
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bronk(set(), set(verts), set())
    return sorted(best)


def _greedy_clique(adj, n, order=None):
    """Greedy maximal clique; returns list."""
    if order is None:
        order = sorted(range(n), key=lambda x: -len(adj[x]))
    clique = []
    cand = set(range(n))
    for v in order:
        if v not in cand:
            continue
        # check v adjacent to all in clique
        if all(v in adj[u] for u in clique):
            clique.append(v)
            cand &= adj[v]
    return clique


def _solve_for_branches(B, adj, n, max_path_len=3):
    """Given branch set B, try to find disjoint interior paths for all pairs.
    Paths may have length 1 (direct edge) or length 2 (one Steiner) or length 3 (two Steiners).
    We first try to assign length-1 or length-2 paths via matching on Steiner vertices.
    For pairs without a direct edge and without a common Steiner, we try length-3.
    
    Strategy:
      - For each pair, determine if direct edge exists.
      - For pairs needing a Steiner, we need a matching from "pairs needing Steiner" to Steiner vertices.
      - If that matching fails, try a more expensive search: allow some pairs to use 2 Steiners.
    
    Returns list of paths (each a list of vertices) or None.
    """
    from itertools import combinations
    Bset = set(B)
    pairs = list(combinations(B, 2))
    m = len(pairs)
    steiners = [v for v in range(n) if v not in Bset]
    Sset = set(steiners)

    # Classify pairs
    direct_pairs = []   # (pair_idx, a, b)
    steiner_pairs = []  # (pair_idx, a, b)
    for i, (a, b) in enumerate(pairs):
        if b in adj[a]:
            direct_pairs.append((i, a, b))
        else:
            steiner_pairs.append((i, a, b))

    # For steiner_pairs, we need to assign distinct Steiner vertices adjacent to both endpoints.
    # Build bipartite graph: left = steiner_pairs, right = steiners
    adjL = []
    for (i, a, b) in steiner_pairs:
        row = []
        for s in steiners:
            if s in adj[a] and s in adj[b]:
                row.append(s)
        adjL.append(row)

    match = _bipartite_max_matching(adjL, n)  # right side is indexed by vertex id 0..n-1, but only steiners matter
    # Actually, match_r is indexed by right vertex id (0..n-1). We need to check that matched vertices are steiners.
    # Let's redo: right side should be steiners only. Let me re-index.
    
    # Re-do with proper indexing
    if steiner_pairs:
        adjL2 = []
        for (i, a, b) in steiner_pairs:
            row = []
            for j, s in enumerate(steiners):
                if s in adj[a] and s in adj[b]:
                    row.append(j)
            adjL2.append(row)
        match2 = _bipartite_max_matching(adjL2, len(steiners))
        if match2 is None:
            # Try length-3 paths for the unmatched ones
            return _solve_with_long_paths(B, adj, n, pairs, steiners, direct_pairs, steiner_pairs)
        # Assign
        paths = [None] * m
        for (i, a, b) in direct_pairs:
            paths[i] = [a, b]
        for idx, (i, a, b) in enumerate(steiner_pairs):
            s_idx = match2[idx]
            s = steiners[s_idx]
            paths[i] = [a, s, b]
        return paths
    else:
        # All direct
        return [[a, b] for (i, a, b) in direct_pairs]


def _solve_with_long_paths(B, adj, n, pairs, steiners, direct_pairs, steiner_pairs):
    """Fallback: allow some pairs to use 2 Steiner vertices (path length 3).
    This is more complex; we do a simple search:
      - First assign direct edges.
      - Then for remaining pairs, try to find vertex-disjoint paths of length 2 or 3.
      - Use backtracking with pruning.
    """
    from itertools import permutations
    Bset = set(B)
    m = len(pairs)
    paths = [None] * m
    used_steiners = set()

    # Assign direct edges first
    for (i, a, b) in direct_pairs:
        paths[i] = [a, b]

    # For steiner_pairs, try to find disjoint paths
    remaining = list(steiner_pairs)
    # Sort by number of available 2-Steiner options (hardest first)
    def count_options(a, b):
        cnt = 0
        for s1 in steiners:
            if s1 in used_steiners:
                continue
            if s1 in adj[a]:
                for s2 in steiners:
                    if s2 == s1 or s2 in used_steiners:
                        continue
                    if s2 in adj[s1] and s2 in adj[b]:
                        cnt += 1
        return cnt

    remaining.sort(key=lambda t: count_options(t[1], t[2]))

    def backtrack(idx):
        if idx == len(remaining):
            return True
        (i, a, b) = remaining[idx]
        # Try length-2 first
        for s in steiners:
            if s in used_steiners:
                continue
            if s in adj[a] and s in adj[b]:
                used_steiners.add(s)
                paths[i] = [a, s, b]
                if backtrack(idx + 1):
                    return True
                used_steiners.remove(s)
        # Try length-3
        for s1 in steiners:
            if s1 in used_steiners or s1 not in adj[a]:
                continue
            for s2 in steiners:
                if s2 == s1 or s2 in used_steiners:
                    continue
                if s2 in adj[s1] and s2 in adj[b]:
                    used_steiners.add(s1)
                    used_steiners.add(s2)
                    paths[i] = [a, s1, s2, b]
                    if backtrack(idx + 1):
                        return True
                    used_steiners.remove(s2)
                    used_steiners.remove(s1)
        return False

    if backtrack(0):
        return paths
    return None


def construct():
    from itertools import combinations

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].add(b)
        adj[b].add(a)

    # Fallback
    u, v = int(edges[0][0]), int(edges[0][1])
    best_branches = [u, v]
    best_paths = [[u, v]]
    best_k = 2

    # Find maximum clique via Bron-Kerbosch
    max_clique = _max_clique_bruteforce(adj, n)
    max_k_possible = len(max_clique)

    # Try from max clique size down
    for k in range(max_k_possible, 1, -1):
        if not _time_left():
            break
        # Generate all k-cliques (subsets of max clique that are cliques)
        # Actually, we need all k-cliques in the graph, not just subsets of one max clique.
        # For small n, we can enumerate combinations and check.
        verts = list(range(n))
        for comb in combinations(verts, k):
            if not _time_left():
                break
            comb_set = set(comb)
            # Check clique
            is_clq = True
            for x, y in combinations(comb, 2):
                if y not in adj[x]:
                    is_clq = False
                    break
            if not is_clq:
                continue
            # Try to solve
            res = _solve_for_branches(list(comb), adj, n)
            if res is not None:
                # Verify disjointness
                interiors = set()
                valid = True
                for p in res:
                    for v in p[1:-1]:
                        if v in interiors or v in comb_set:
                            valid = False
                            break
                        interiors.add(v)
                    if not valid:
                        break
                if valid:
                    best_k = k
                    best_branches = [int(x) for x in comb]
                    best_paths = [[int(x) for x in p] for p in res]
                    break  # Found a valid k-clique, move to next k
        # If we found a solution for this k, we can stop since we're going descending
        if best_k == k:
            break

    if len(best_branches) < 2:
        best_branches = [u, v]
        best_paths = [[u, v]]

    best_branches = [int(x) for x in best_branches]
    best_paths = [[int(x) for x in p] for p in best_paths]

    return {'branches': best_branches, 'paths': best_paths}
# EVOLVE-BLOCK-END
