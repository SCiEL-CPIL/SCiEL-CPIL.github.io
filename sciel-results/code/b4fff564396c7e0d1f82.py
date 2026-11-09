import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    adj_list = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    def bfs_path(s, t, forbidden):
        if s == t:
            return [s]
        if t in adj[s]:
            return [s, t]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj_list[cur]:
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

    def solve_disjoint_paths(branches, max_restarts=100):
        k = len(branches)
        if k < 2:
            return None
        branches_set = set(branches)
        pairs = [(branches[i], branches[j]) for i in range(k) for j in range(i+1, k)]
        num_pairs = len(pairs)
        import random

        for restart in range(max_restarts):
            if not time_ok():
                return None
            if restart == 0:
                pair_order = list(pairs)
            elif restart == 1:
                pair_order = sorted(pairs, key=lambda p: (len(bfs_path(p[0], p[1], branches_set - set(p)), 1000)))
            else:
                pair_order = list(pairs)
                random.shuffle(pair_order)

            used_interior = set()
            path_map = {}
            success = True

            for u, v in pair_order:
                a, b = min(u, v), max(u, v)
                key = (a, b)
                forbidden = branches_set - {u, v}
                forbidden_with_used = forbidden | used_interior
                p = bfs_path(u, v, forbidden_with_used)
                if p is None:
                    success = False
                    break
                interior = set(p[1:-1])
                if interior & used_interior:
                    success = False
                    break
                if interior & (branches_set - {u, v}):
                    success = False
                    break
                path_map[key] = p
                used_interior |= interior

            if success and len(path_map) == num_pairs:
                output_paths = []
                for i in range(k):
                    for j in range(i+1, k):
                        a, b = branches[i], branches[j]
                        key = (min(a, b), max(a, b))
                        p = path_map[key]
                        if p[0] == branches[i]:
                            output_paths.append(p)
                        else:
                            output_paths.append(p[::-1])
                return output_paths

        return None

    def find_max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if not time_ok():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if Pset:
                pivot_candidates = list(Pset | X)
            else:
                pivot_candidates = list(X)
            if not pivot_candidates:
                return
            u = max(pivot_candidates, key=lambda v: len(adj[v] & Pset))
            candidates = [v for v in Pset if v not in adj[u]]
            candidates.sort(key=lambda x: -degrees[x])
            for v in candidates:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}
        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    max_clique = find_max_clique()
    best_result = None
    best_k = 0
    clique_list = sorted(max_clique)

    if len(clique_list) >= 2:
        paths = solve_disjoint_paths(clique_list, max_restarts=100)
        if paths is not None:
            best_result = (clique_list, paths)
            best_k = len(clique_list)

    if best_k < len(clique_list):
        for size in range(len(clique_list) - 1, 1, -1):
            if not time_ok():
                break
            found = False
            import random
            for attempt in range(min(50, 2**size)):
                if not time_ok():
                    break
                subset = list(random.sample(clique_list, size))
                subset.sort()
                paths = solve_disjoint_paths(subset, max_restarts=50)
                if paths is not None:
                    best_result = (subset, paths)
                    best_k = size
                    found = True
                    break
            if found:
                break
            if len(clique_list) - size <= 3 and size >= 2:
                for remove_set in combinations(clique_list, len(clique_list) - size):
                    if not time_ok():
                        break
                    subset = [v for v in clique_list if v not in remove_set]
                    subset.sort()
                    paths = solve_disjoint_paths(subset, max_restarts=30)
                    if paths is not None:
                        best_result = (subset, paths)
                        best_k = size
                        found = True
                        break
                if found:
                    break

    if best_result is not None:
        current_branches = list(best_result[0])
        current_paths = best_result[1]
        improved = True
        while improved and time_ok():
            improved = False
            current_set = set(current_branches)
            candidates = []
            for v in range(n):
                if v in current_set:
                    continue
                score = len(adj[v] & current_set)
                candidates.append((score, v))
            candidates.sort(key=lambda x: -x[0])
            for score, v in candidates:
                if score == 0:
                    break
                new_branches = sorted(current_branches + [v])
                new_paths = solve_disjoint_paths(new_branches, max_restarts=50)
                if new_paths is not None:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_result = (current_branches, new_paths)
                    best_k = len(current_branches)
                    improved = True
                    break

    if best_k < 3:
        high_deg_vertices = sorted(range(n), key=lambda x: -degrees[x])[:20]
        for i in range(len(high_deg_vertices)):
            for j in range(i+1, len(high_deg_vertices)):
                if not time_ok():
                    break
                pair = [high_deg_vertices[i], high_deg_vertices[j]]
                paths = solve_disjoint_paths(pair, max_restarts=10)
                if paths is not None:
                    if best_k < 2:
                        best_result = (pair, paths)
                        best_k = 2
                        break
            if best_k >= 2:
                break

    if best_result is None or best_k < 2:
        u, v = edges[0]
        best_result = ([int(u), int(v)], [[int(u), int(v)]])
        best_k = 2

    branches = [int(b) for b in best_result[0]]
    paths_out = [[int(v) for v in p] for p in best_result[1]]

    return {
        'branches': branches,
        'paths': paths_out
    }
