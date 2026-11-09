import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # ---------- Bron-Kerbosch with pivot to find max clique ----------
    best_clique = []

    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Pivot
        candidates_for_pivot = list(Pset) + list(X)
        u = None
        max_neighbors = -1
        for v in candidates_for_pivot:
            count = len(adj[v] & Pset)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        if u is None:
            candidates = list(Pset)
        else:
            candidates = [v for v in Pset if v not in adj[u]]
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        for v in candidates:
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset = Pset - {v}
            X = X | {v}

    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())

    if not best_clique:
        best_clique = [0, 1]

    # ---------- Fast path finding (direct, length-2, length-3, BFS) ----------
    def _find_path_length2(u, v, forbidden):
        """Find a path of length exactly 2: u - w - v."""
        for w in adj[u]:
            if w == v or w in forbidden:
                continue
            if v in adj[w]:
                return [u, w, v]
        return None

    def _find_path_length3(u, v, forbidden):
        """Find a path of length exactly 3: u - a - b - v."""
        for a in adj[u]:
            if a == v or a in forbidden:
                continue
            for b in adj[a]:
                if b == u or b == v or b in forbidden:
                    continue
                if v in adj[b]:
                    return [u, a, b, v]
        return None

    def shortest_path(s, t, forbidden):
        """BFS for shortest path from s to t avoiding forbidden set (interior only)."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nb in forbidden:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path(u, v, forbidden, max_len=3):
        """Find a simple path from u to v with length <= max_len, avoiding forbidden vertices."""
        # Direct edge
        if v in adj[u]:
            return [u, v]
        if max_len >= 2:
            path = _find_path_length2(u, v, forbidden)
            if path:
                return path
        if max_len >= 3:
            path = _find_path_length3(u, v, forbidden)
            if path:
                return path
        # Fall back to BFS
        return shortest_path(u, v, forbidden)

    # ---------- Verify a clique subdivision using backtracking ----------
    def verify_clique_subdivision(clique):
        """Try to find internally disjoint paths for all pairs in clique.
        Returns (True, paths_dict) or (False, None)."""
        k = len(clique)
        if k < 2:
            return False, None
        branches_set = set(clique)
        pairs = list(combinations(range(k), 2))

        # Quick check: every pair must have at least one path
        for i, j in pairs:
            s, t = clique[i], clique[j]
            forbidden = branches_set - {s, t}
            if find_path(s, t, forbidden, max_len=3) is None:
                return False, None

        # Sort pairs by difficulty (fewer options first)
        pair_difficulty = []
        for i, j in pairs:
            s, t = clique[i], clique[j]
            forbidden = branches_set - {s, t}
            # Count available intermediate vertices for length-2 paths
            count = 0
            if t in adj[s]:
                count += 1
            for w in adj[s]:
                if w != t and w not in forbidden and t in adj[w]:
                    count += 1
            pair_difficulty.append((count, (i, j)))
        pair_difficulty.sort(key=lambda x: x[0])
        sorted_pairs = [p for _, p in pair_difficulty]

        # Backtracking search
        best_paths = [None] * len(sorted_pairs)
        search_deadline = min(time.time() + 5.0, deadline)

        def backtrack(idx, used_interior):
            if time.time() > search_deadline:
                return False
            if idx == len(sorted_pairs):
                return True
            i, j = sorted_pairs[idx]
            s, t = clique[i], clique[j]
            forbidden = branches_set - {s, t}
            forbidden = forbidden | used_interior

            # Try paths of increasing length
            for max_len in [1, 2, 3]:
                path = find_path(s, t, forbidden, max_len)
                if path:
                    interior = set(path[1:-1])
                    if not interior & used_interior:
                        best_paths[idx] = path
                        if backtrack(idx + 1, used_interior | interior):
                            return True
                        best_paths[idx] = None
            return False

        if backtrack(0, set()):
            paths_dict = {}
            for idx, (i, j) in enumerate(sorted_pairs):
                paths_dict[(i, j)] = best_paths[idx]
            return True, paths_dict
        return False, None

    # ---------- Search for maximum valid subdivision ----------
    best_valid = None
    best_valid_size = 0

    # Try the max clique
    if len(best_clique) >= 2:
        valid, paths = verify_clique_subdivision(best_clique)
        if valid:
            best_valid = (best_clique, paths)
            best_valid_size = len(best_clique)

    # Try subsets of the clique
    if best_valid is None:
        for size in range(len(best_clique) - 1, 1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in combinations(best_clique, size):
                valid, paths = verify_clique_subdivision(list(subset))
                if valid:
                    best_valid = (list(subset), paths)
                    best_valid_size = size
                    found = True
                    break
            if found:
                break

    # Try to grow: add vertices one by one
    if best_valid is not None:
        current_branches = list(best_valid[0])
        current_paths = best_valid[1]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for v in range(n):
                if v in set(current_branches):
                    continue
                new_branches = current_branches + [v]
                valid, new_paths = verify_clique_subdivision(new_branches)
                if valid:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_valid = (current_branches, current_paths)
                    best_valid_size = len(current_branches)
                    improved = True
                    break

    # Fallback
    if best_valid is None or len(best_valid[0]) < 2:
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j)) or paths_dict.get((j, i))
            if p is None:
                p = [branches[i], branches[j]]
            if p[0] != branches[i]:
                p = p[::-1]
            output_paths.append(list(p))

    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
