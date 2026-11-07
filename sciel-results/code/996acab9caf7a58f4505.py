import time
import random
from itertools import combinations
from collections import deque
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 75.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_sorted = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def validate(branches, paths):
        bset = set(branches)
        if len(branches) < 2:
            return 0
        if len(branches) != len(bset):
            return 0
        expected_paths = len(branches) * (len(branches) - 1) // 2
        if len(paths) != expected_paths:
            return 0
        used_interiors = set()
        path_pairs = set()
        for p in paths:
            if len(p) < 2:
                return 0
            u, v = p[0], p[-1]
            if u not in bset or v not in bset:
                return 0
            if u == v:
                return 0
            seen = set()
            for node in p:
                if node in seen:
                    return 0
                seen.add(node)
            for node in p[1:-1]:
                if node in bset:
                    return 0
                if node in used_interiors:
                    return 0
            used_interiors.update(p[1:-1])
            for i in range(len(p) - 1):
                if p[i + 1] not in adj[p[i]]:
                    return 0
            path_pairs.add(frozenset((u, v)))
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                if frozenset((branches[i], branches[j])) not in path_pairs:
                    return 0
        return len(branches)

    def _find_path_bfs(src, dst, forbidden, adj_sorted):
        if src == dst:
            return [src]
        if dst in adj[src]:
            return [src, dst]
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in adj_sorted[cur]:
                if nb in parent:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                parent[nb] = cur
                if nb == dst:
                    path = [dst]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(nb)
        return None

    def _find_path_bfs_shortest(src, dst, forbidden, adj_sorted):
        """BFS that prefers shorter paths by exploring in order of distance from src."""
        if src == dst:
            return [src]
        if dst in adj[src]:
            return [src, dst]
        parent = {src: None}
        dist = {src: 0}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nb in adj_sorted[cur]:
                if nb in parent:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                parent[nb] = cur
                dist[nb] = dist[cur] + 1
                if nb == dst:
                    path = [dst]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(nb)
        return None

    def _find_path_random(src, dst, forbidden, rng, adj_sorted, max_depth=14):
        if src == dst:
            return [src]
        if dst in adj[src]:
            return [src, dst]

        def dfs(cur, path, visited):
            if len(path) > max_depth + 2:
                return None
            if cur == dst:
                return list(path)
            nbrs = list(adj_sorted[cur])
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb in visited:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
            return None

        visited = {src}
        return dfs(src, [src], visited)

    def find_paths_for_branches(branches, rng, max_attempts=300):
        k = len(branches)
        if k < 2:
            return None

        bset = set(branches)
        pair_list = list(combinations(branches, 2))

        for attempt in range(max_attempts):
            if time_up():
                return None

            paths_dict = {}
            used_interiors = set()
            success = True

            # Order: direct edges first, then randomize
            edge_pairs = [p for p in pair_list if p[1] in adj[p[0]]]
            non_edge_pairs = [p for p in pair_list if p[1] not in adj[p[0]]]
            rng.shuffle(non_edge_pairs)

            # Strategy: use different ordering heuristics based on attempt
            if attempt % 5 == 0:
                # Sort by degree sum ascending (easy pairs first)
                non_edge_pairs.sort(key=lambda p: degrees[p[0]] + degrees[p[1]])
            elif attempt % 5 == 1:
                # Sort by degree sum descending (hard pairs first)
                non_edge_pairs.sort(key=lambda p: degrees[p[0]] + degrees[p[1]], reverse=True)
            elif attempt % 5 == 2:
                # Sort by minimum degree in pair
                non_edge_pairs.sort(key=lambda p: min(degrees[p[0]], degrees[p[1]]))
            elif attempt % 5 == 3:
                # Random order
                pass  # already shuffled
            else:
                # Sort by average degree
                non_edge_pairs.sort(key=lambda p: (degrees[p[0]] + degrees[p[1]]) / 2)

            ordered = edge_pairs + non_edge_pairs

            for (u, v) in ordered:
                if time_up():
                    return None
                forbidden = set(used_interiors)
                forbidden.update(bset)
                forbidden.discard(u)
                forbidden.discard(v)

                # Try BFS first, then randomized DFS
                path = _find_path_bfs(u, v, forbidden, adj_sorted)
                if path is None:
                    path = _find_path_random(u, v, forbidden, rng, adj_sorted)

                if path is None:
                    success = False
                    break

                interior = set(path[1:-1])
                if interior & bset or interior & used_interiors:
                    success = False
                    break

                used_interiors |= interior
                paths_dict[frozenset((u, v))] = path

            if success:
                return [paths_dict[frozenset((u, v))] for (u, v) in pair_list]

        return None

    def solve_ilp(k):
        if not HAS_MILP:
            return None
        m = len(edges)
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        c = np.array([1] * n + [0] * m)

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        row = np.zeros(n + m)
        for v in range(n):
            row[v] = 1
        A_eq.append(row)
        b_eq.append(k)

        for v in range(n):
            incident_edges = []
            for (u, w), idx in edge_idx.items():
                if u == v or w == v:
                    incident_edges.append(idx)

            row = np.zeros(n + m)
            row[v] = k - 1
            for e in incident_edges:
                row[n + e] = -1
            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(n + m)
            for e in incident_edges:
                row[n + e] = 1
            row[v] = -(k - 3)
            A_ub.append(row)
            b_ub.append(2)

        constraints = [
            LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)),
            LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq))
        ]
        bounds = Bounds(lb=[0] * (n + m), ub=[1] * (n + m))
        integrality = np.ones(n + m)

        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                b_vals = res.x[:n]
                branches = [int(v) for v in range(n) if b_vals[v] > 0.5]
                return branches
            else:
                return None
        except Exception:
            return None

    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Phase 1: ILP for k from high to low
    max_k = 10
    for k in range(max_k, 1, -1):
        if time_up():
            break
        branches = solve_ilp(k)
        if branches and len(branches) >= k:
            paths = find_paths_for_branches(branches, rng, max_attempts=200)
            if paths:
                cnt = validate(branches, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (branches, paths)
                    break

    # Phase 2: Greedy with diverse initial sets
    initial_sets = [
        sorted_verts[:11],
        sorted_verts[:10],
        sorted_verts[:9],
        sorted_verts[:8],
        [0, 1, 3, 5, 20, 22, 12, 4],
        [0, 1, 3, 5, 9, 20, 22, 4],
        [0, 1, 3, 5, 20, 22, 12, 9],
        [0, 1, 3, 5, 20, 22, 4, 9],
        [0, 1, 3, 5, 20, 22, 12, 4, 9],
        [0, 1, 3, 5, 20, 22, 12, 4, 9, 15],
    ]
    for init in initial_sets:
        if time_up():
            break
        paths = find_paths_for_branches(init, rng, max_attempts=400)
        if paths:
            cnt = validate(init, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (init, paths)

    # Phase 3: Greedy growth from best
    if best_result:
        current = list(best_result[0])
        for _ in range(20):
            if time_up():
                break
            added = False
            candidates = [v for v in range(n) if v not in set(current)]
            candidates.sort(key=lambda v: degrees[v], reverse=True)
            for v in candidates:
                if time_up():
                    break
                new_branches = current + [v]
                new_paths = find_paths_for_branches(new_branches, rng, max_attempts=300)
                if new_paths:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        added = True
                        break
            if not added:
                break

    # Phase 4: Randomized restarts with random branch sets
    for trial in range(40):
        if time_up():
            break
        # Pick a random subset of high-degree vertices
        pool = sorted_verts[:14]
        size = random.Random(trial * 7 + 1).choice([7, 8, 9, 10, 11])
        candidate = random.Random(trial * 13 + 3).sample(pool, min(size, len(pool)))
        paths = find_paths_for_branches(candidate, rng, max_attempts=250)
        if paths:
            cnt = validate(candidate, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (candidate, paths)

    if best_result is None:
        branches = [0, 1]
        paths = [[0, 1]]
        best_result = (branches, paths)

    branches, paths = best_result
    branch_list = [int(b) for b in sorted(branches)]
    # Reorder paths to match sorted branch pairs
    sorted_pairs = list(combinations(sorted(branches), 2))
    path_map = {}
    for p in paths:
        path_map[frozenset((p[0], p[-1]))] = p
    path_list = []
    for a, b in sorted_pairs:
        p = path_map.get(frozenset((a, b)))
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
