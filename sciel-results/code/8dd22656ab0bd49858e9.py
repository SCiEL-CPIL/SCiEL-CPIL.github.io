import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges, n):
    """Build adjacency list from base path + added edges."""
    adj = [set() for _ in range(n)]
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj, n):
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    return False
    return True

def _max_diameter(adj, n, limit=4):
    """Return max distance, or limit+1 if any pair exceeds limit."""
    maxd = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= limit:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    if dist[v] > maxd:
                        maxd = dist[v]
        if any(d == -1 or d > limit for d in dist):
            return limit + 1
    return maxd

def _check_valid(added_edges, n):
    adj = _build_adj(added_edges, n)
    if not _is_triangle_free(adj, n):
        return False
    return _max_diameter(adj, n, limit=4) <= 4

def construct():
    n = 24
    deadline = time.time() + 85  # safety margin before 90s kill

    # Precompute bipartite candidate edges (even-odd, not in path)
    candidates = []
    for u in range(0, n, 2):
        for v in range(1, n, 2):
            if abs(u - v) > 1:  # not already a path edge
                candidates.append((u, v))

    # Greedy: build a good solution first
    greedy_edges = []
    current_adj = _build_adj([], n)
    for _ in range(10):
        if time.time() > deadline:
            break
        diam = _max_diameter(current_adj, n, limit=4)
        if diam <= 4:
            break
        # Find pair with max distance
        maxd = 0
        worst_pair = (0, n - 1)
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in current_adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > maxd:
                            maxd = dist[v]
                            worst_pair = (s, v)
        # Try all candidate edges, pick best reduction
        best_edge = None
        best_diam = maxd
        for e in candidates:
            if e in greedy_edges:
                continue
            new_edges = greedy_edges + [e]
            new_adj = _build_adj(new_edges, n)
            d = _max_diameter(new_adj, n, limit=4)
            if d < best_diam:
                best_diam = d
                best_edge = e
        if best_edge is None:
            break
        greedy_edges.append(best_edge)
        current_adj = _build_adj(greedy_edges, n)

    greedy_sol = greedy_edges if _check_valid(greedy_edges, n) else None

    # Now search for minimal k
    best_found = greedy_sol
    for k in range(1, len(candidates) + 1):
        if time.time() > deadline:
            break
        if best_found is not None and k >= len(best_found):
            break

        found = None
        if k <= 3:
            # Exhaustive for small k
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _check_valid(list(combo), n):
                    found = list(combo)
                    break
        else:
            # Local search: start from greedy, try to remove/swap
            if best_found is not None:
                # Try to reduce greedy solution
                current = list(best_found)
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(current)):
                        trial = current[:i] + current[i+1:]
                        if _check_valid(trial, n):
                            current = trial
                            improved = True
                            break
                    if not improved:
                        # Try swaps
                        for i in range(len(current)):
                            for j in range(len(current)):
                                if i == j:
                                    continue
                                for e in candidates:
                                    if e in current:
                                        continue
                                    trial = current[:i] + [e] + current[i+1:]
                                    if _check_valid(trial, n):
                                        current = trial
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                if len(current) < k:
                    found = current
                elif len(current) == k:
                    found = current

        if found is not None:
            return [[int(u), int(v)] for u, v in found]

    # Fallback
    if best_found is not None:
        return [[int(u), int(v)] for u, v in best_found]
    # Hardcoded valid solution: 6 edges
    sol = [(12, 1), (12, 5), (12, 9), (12, 15), (12, 19), (12, 23)]
    if _check_valid(sol, n):
        return [[int(u), int(v)] for u, v in sol]
    return [[int(u), int(v)] for u, v in [(11, 1), (11, 4), (11, 7), (11, 15), (11, 18), (11, 21), (11, 23)]]
