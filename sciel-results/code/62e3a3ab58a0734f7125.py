"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest paths
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src][v] == -1:
                    dist[src][v] = dist[src][u] + 1
                    q.append(v)

    # Bron-Kerbosch with pivot to find maximum clique
    def find_max_clique():
        best = []

        def bronk(R, P_set, X):
            nonlocal best
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P_set:
                return
            pivot_candidates = list(P_set | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
                candidates = list(P_set - adj[pivot])
            else:
                candidates = list(P_set)
            for v in candidates:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}

        bronk(set(), set(range(n)), set())
        return best

    max_clique = find_max_clique()

    def find_shortest_path(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden internal vertices."""
        if u == v:
            return None
        visited = {u}
        parent = {u: None}
        q = deque([u])
        while q:
            curr = q.popleft()
            if curr == v:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[curr]:
                if nb not in visited and (nb == v or nb not in forbidden):
                    visited.add(nb)
                    parent[nb] = curr
                    q.append(nb)
        return None

    def try_subdivision(branches):
        """Try to find internally-disjoint paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if impossible."""
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        # Order pairs by "difficulty" (longest shortest path first)
        pair_info = []
        for u, v in pairs:
            d = dist[u][v]
            if d == -1:
                return None
            pair_info.append((d, u, v))
        pair_info.sort(key=lambda x: -x[0])

        paths = {}
        used_int = set()

        def backtrack(idx):
            if idx == len(pair_info):
                return True
            d, u, v = pair_info[idx]
            forbidden = used_int | (branch_set - {u, v})
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return False
            interiors = set(path[1:-1])
            if interiors & used_int:
                return False
            paths[(u, v)] = path
            used_int.update(interiors)
            if backtrack(idx + 1):
                return True
            del paths[(u, v)]
            used_int.difference_update(interiors)
            return False

        if backtrack(0):
            return paths
        return None

    # Start with the max clique
    best_result = None
    if len(max_clique) >= 2:
        branches = sorted(max_clique)
        result = try_subdivision(branches)
        if result is not None:
            best_result = (branches, result)

    # Try to extend: add vertices one at a time
    if best_result is not None:
        current_branches = set(best_result[0])
        for extra in range(n):
            if extra in current_branches:
                continue
            candidate = sorted(current_branches | {extra})
            result = try_subdivision(candidate)
            if result is not None:
                best_result = (candidate, result)
                current_branches = set(candidate)

    # Try random sets of various sizes to find even larger subdivisions
    import random
    random.seed(123)

    current_k = len(best_result[0]) if best_result else 2

    # Try to grow from the current best
    for _ in range(300):
        if time.time() > deadline:
            break
        current_branches = set(best_result[0])
        for extra in random.sample(range(n), min(n, 10)):
            if extra in current_branches:
                continue
            candidate = sorted(current_branches | {extra})
            result = try_subdivision(candidate)
            if result is not None:
                best_result = (candidate, result)
                current_branches = set(candidate)
                break

    # Try random subsets of increasing size
    for size in range(current_k + 1, min(n + 1, current_k + 8)):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(5000):
            if time.time() > deadline:
                break
            candidates = random.sample(range(n), size)
            result = try_subdivision(candidates)
            if result is not None:
                best_result = (sorted(candidates), result)
                current_k = size
                found = True
                break
        if not found:
            continue

    # Final: try to grow the best result further
    for _ in range(200):
        if time.time() > deadline:
            break
        current_branches = set(best_result[0])
        improved = False
        for extra in random.sample(range(n), min(n, 15)):
            if extra in current_branches:
                continue
            candidate = sorted(current_branches | {extra})
            result = try_subdivision(candidate)
            if result is not None:
                best_result = (candidate, result)
                current_branches = set(candidate)
                improved = True
                break
        if not improved:
            break

    branches, paths = best_result
    branches_out = [int(b) for b in branches]
    paths_out = []
    for (u, v), path in sorted(paths.items()):
        paths_out.append([int(x) for x in path])

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
