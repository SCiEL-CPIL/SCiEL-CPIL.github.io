import time
import random
from collections import deque
from itertools import combinations
import networkx as nx

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

    # Greedy + random restarts to find good cliques
    def greedy_clique(start_order):
        remaining = set(start_order)
        clique = set()
        while remaining:
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.add(best_v)
            remaining = remaining & adj[best_v]
        return clique

    for trial in range(100):
        if time.time() > deadline - 30:
            break
        shuffled = list(range(n))
        random.shuffle(shuffled)
        c = greedy_clique(shuffled)
        if len(c) > len(best_clique):
            best_clique = list(c)

    if not best_clique:
        best_clique = [0, 1]

    # ---------- Path finding (shortest path avoiding forbidden vertices) ----------
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
                    # Reconstruct
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

    # ---------- Verify a clique subdivision ----------
    def verify_clique_subdivision(clique):
        """Try to find internally disjoint paths for all pairs in clique.
        Returns (True, paths_dict) or (False, None)."""
        k = len(clique)
        if k < 2:
            return False, None
        branches_set = set(clique)
        pairs = list(combinations(range(k), 2))
        # Order pairs: try to assign paths greedily, shortest first
        # For each pair, compute shortest path avoiding all branches except endpoints
        # Then check disjointness; if conflict, try alternative paths

        # Strategy: for each pair, find a path that avoids all previously used interior vertices
        # Order pairs by "difficulty" (shorter available paths first)
        # Compute initial shortest paths
        pair_paths = {}
        for i, j in pairs:
            s, t = clique[i], clique[j]
            forbidden = branches_set - {s, t}
            p = shortest_path(s, t, forbidden)
            if p is None:
                return False, None
            pair_paths[(i, j)] = p

        # Check disjointness
        def check_disjoint(paths):
            used_interior = set()
            for key, p in paths.items():
                interior = set(p[1:-1])
                if interior & used_interior:
                    return False
                used_interior |= interior
            return True

        if check_disjoint(pair_paths):
            return True, pair_paths

        # Try to resolve conflicts by finding alternative paths
        # Use a more sophisticated approach: iterative reassignment
        # For simplicity, try multiple random orderings of path assignment
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Shuffle pairs and assign greedily
            shuffled_pairs = list(pairs)
            random.shuffle(shuffled_pairs)
            assigned = {}
            used_interior = set()
            success = True
            for i, j in shuffled_pairs:
                s, t = clique[i], clique[j]
                forbidden = (branches_set - {s, t}) | used_interior
                p = shortest_path(s, t, forbidden)
                if p is None:
                    success = False
                    break
                assigned[(i, j)] = p
                used_interior |= set(p[1:-1])
            if success:
                # Verify all pairs are covered
                if len(assigned) == len(pairs):
                    # Build final pair_paths in consistent order
                    final_paths = {}
                    for i, j in pairs:
                        final_paths[(i, j)] = assigned[(i, j)]
                    return True, final_paths
        return False, None

    # ---------- Search for maximum valid subdivision ----------
    # Start with the max clique found, then try to grow or shrink
    best_valid = None
    best_valid_size = 0

    # Try the max clique
    if len(best_clique) >= 2:
        valid, paths = verify_clique_subdivision(best_clique)
        if valid:
            best_valid = (best_clique, paths)
            best_valid_size = len(best_clique)

    # Try subsets of the clique (in case full clique doesn't work)
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
        while improved and time.time() < deadline:
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

    # Also try greedy construction from scratch with verification
    # Build up a valid subdivision by adding vertices that maintain validity
    if best_valid is None or len(best_valid[0]) < 2:
        # Fallback: find any valid pair
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        # Absolute fallback
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    # Ensure branches are sorted for consistent output
    branch_to_idx = {b: i for i, b in enumerate(branches)}

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j)) or paths_dict.get((j, i))
            if p is None:
                # Should not happen if verification passed
                p = [branches[i], branches[j]]
            # Ensure path starts and ends with correct branches
            if p[0] != branches[i] and p[-1] != branches[j]:
                p = p[::-1]
            if p[0] != branches[i]:
                # Swap if needed
                p = p[::-1]
            output_paths.append(list(p))

    # Final type safety
    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
