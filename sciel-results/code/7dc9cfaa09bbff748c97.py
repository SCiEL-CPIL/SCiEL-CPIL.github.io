"""Combined Gallai path cover: vertex-start greedy + local search + Eulerian augmentation."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    _start = time.time()
    _DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def validate(paths, edge_set):
        covered = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        return covered == edge_set

    def _bfs(src, adj):
        d = {src: 0}
        par = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in d:
                    d[nb] = d[cur] + 1
                    par[nb] = cur
                    q.append(nb)
        return d, par

    def _shortest_path(src, dst, adj):
        if src == dst:
            return [src]
        d, par = _bfs(src, adj)
        if dst not in d:
            return None
        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = par[node]
        path.reverse()
        return path

    def _shortest_path_avoiding(src, dst, adj, avoid_set):
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb != dst and nb in avoid_set:
                    continue
                visited.add(nb)
                parent[nb] = cur
                if nb == dst:
                    path = []
                    node = dst
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                q.append(nb)
        return None

    def _min_weight_perfect_matching(vertices, dist):
        n = len(vertices)
        if n == 0:
            return []
        if n % 2 != 0:
            return None

        if n <= 16:
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(mask):
                if mask == (1 << n) - 1:
                    return 0, []

                i = -1
                for j in range(n):
                    if not (mask >> j) & 1:
                        i = j
                        break
                if i == -1:
                    return 0, []

                best = float('inf')
                best_j = -1
                best_rest = None

                for j in range(i + 1, n):
                    if not (mask >> j) & 1:
                        u, v = vertices[i], vertices[j]
                        key = (u, v) if u < v else (v, u)
                        w = dist.get(key, float('inf'))
                        if w < float('inf'):
                            new_mask = mask | (1 << i) | (1 << j)
                            rc, rp = dp(new_mask)
                            total = w + rc
                            if total < best:
                                best = total
                                best_j = j
                                best_rest = rp

                if best_j == -1:
                    return float('inf'), []

                return best, [(vertices[i], vertices[best_j])] + best_rest

            cost, pairs = dp(0)
            if cost >= float('inf'):
                return None
            return pairs
        else:
            all_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    u, v = vertices[i], vertices[j]
                    key = (u, v) if u < v else (v, u)
                    w = dist.get(key, float('inf'))
                    if w < float('inf'):
                        all_edges.append((w, u, v))
            all_edges.sort()

            used = set()
            pairs = []
            for w, u, v in all_edges:
                if u not in used and v not in used:
                    pairs.append((u, v))
                    used.add(u)
                    used.add(v)
                    if len(used) == n:
                        break
            return pairs

    def _find_eulerian_circuit_from_multiset(edge_list):
        if not edge_list:
            return []

        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            adj[u].append((v, idx))
            adj[v].append((u, idx))

        if not adj:
            return []

        start = next(iter(adj))
        stack = [start]
        circuit = []
        used = [False] * len(edge_list)

        while stack:
            v = stack[-1]
            found = False
            while adj[v]:
                nb, idx = adj[v].pop()
                if not used[idx]:
                    used[idx] = True
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())

        circuit.reverse()

        if len(circuit) != len(edge_list) + 1:
            return None

        return circuit

    def _greedy_path_cover_vertex_start(edge_set, adj, n, rng):
        """Vertex-start greedy with minimum-degree heuristic, biased toward odd-degree vertices."""
        if not edge_set:
            return []

        remaining = set(edge_set)
        paths = []

        # Precompute full degrees
        full_deg = defaultdict(int)
        for (u, v) in edge_set:
            full_deg[u] += 1
            full_deg[v] += 1
        odd_vertices = [v for v in range(n) if full_deg[v] % 2 == 1]

        while remaining:
            # Compute remaining degrees
            rem_deg = defaultdict(int)
            for (u, v) in remaining:
                rem_deg[u] += 1
                rem_deg[v] += 1

            # Pick start vertex: bias toward odd-degree in remaining graph
            rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
            if rem_odd and rng.random() < 0.8:
                start_v = rng.choice(rem_odd)
            elif rem_deg:
                start_v = rng.choice([v for v in rem_deg if rem_deg[v] > 0])
            else:
                break

            # Find long path from start_v
            path = [start_v]
            visited = {start_v}
            current = start_v

            while True:
                # Find available neighbors
                available = []
                for nb in adj[current]:
                    if nb not in visited:
                        key1 = (current, nb) if current <= nb else (nb, current)
                        if key1 in remaining:
                            available.append(nb)

                if not available:
                    break

                # Heuristic: choose neighbor with minimum remaining degree
                best_nb = None
                min_deg = float('inf')
                for nb in available:
                    deg = rem_deg[nb]
                    if deg < min_deg:
                        min_deg = deg
                        best_nb = nb

                # Randomize slightly
                if rng.random() < 0.15 and len(available) > 1:
                    best_nb = rng.choice(available)

                # Move
                u, v = current, best_nb
                key = (u, v) if u <= v else (v, u)
                remaining.discard(key)

                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb

            if len(path) >= 2:
                paths.append(path)
            else:
                # Single vertex, take one edge
                for (u, v) in remaining:
                    if u == start_v:
                        path = [start_v, v]
                        remaining.discard((u, v) if u <= v else (v, u))
                        break
                    elif v == start_v:
                        path = [start_v, u]
                        remaining.discard((u, v) if u <= v else (v, u))
                        break
                if len(path) >= 2:
                    paths.append(path)

        return paths

    def _local_search_merge(paths, edge_set, adj, time_limit=5.0):
        """Try to merge paths by connecting endpoints with graph edges."""
        start_time = time.time()
        best = [list(p) for p in paths]

        def is_simple(p):
            return len(set(p)) == len(p)

        while time.time() - start_time < time_limit:
            improved = False
            n = len(best)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = best[i], best[j]

                    candidates = []
                    for a, b in [(p1[-1], p2[0]), (p1[0], p2[0]),
                                  (p1[-1], p2[-1]), (p1[0], p2[-1])]:
                        key = (a, b) if a <= b else (b, a)
                        if key in edge_set:
                            if a == p1[-1] and b == p2[0]:
                                merged = p1 + p2[1:]
                            elif a == p1[0] and b == p2[0]:
                                merged = list(reversed(p1)) + p2[1:]
                            elif a == p1[-1] and b == p2[-1]:
                                merged = p1 + list(reversed(p2[1:]))
                            else:
                                merged = list(reversed(p1)) + list(reversed(p2[1:]))

                            if is_simple(merged):
                                candidates.append(merged)

                    if candidates:
                        merged = candidates[0]
                        new_paths = [best[k] for k in range(n) if k != i and k != j]
                        new_paths.append(merged)
                        best = new_paths
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return best

    def _eulerian_augmentation(edge_set, adj, n, time_budget=10.0):
        """Eulerian augmentation with vertex-disjoint path enforcement."""
        if not edge_set:
            return []

        odd_vertices = [v for v in range(n) if sum(1 for (u, w) in edge_set if u == v or w == v) % 2 == 1]

        if not odd_vertices:
            return None

        n_odd = len(odd_vertices)
        if n_odd % 2 != 0:
            return None

        # BFS distances
        dist_matrix = {}
        for src in odd_vertices:
            d, _ = _bfs(src, adj)
            for dst in odd_vertices:
                if dst != src and dst in d:
                    key = (src, dst) if src < dst else (dst, src)
                    dist_matrix[key] = d[dst]

        pairs = _min_weight_perfect_matching(odd_vertices, dist_matrix)
        if not pairs:
            return None

        # Try to find vertex-disjoint paths for each pair
        best_result = None
        for attempt in range(30):
            if time.time() > _start + time_budget:
                break

            rng = random.Random(attempt)
            shuffled_pairs = list(pairs)
            rng.shuffle(shuffled_pairs)

            used_vertices = set()
            aug_paths = []
            success = True

            for (a, b) in shuffled_pairs:
                path = _shortest_path_avoiding(a, b, adj, used_vertices)
                if path is None:
                    path = _shortest_path(a, b, adj)
                    if path is None:
                        success = False
                        break
                    internal = set(path[1:-1])
                    if internal & used_vertices:
                        success = False
                        break

                internal = set(path[1:-1])
                if internal & used_vertices:
                    success = False
                    break

                used_vertices |= internal
                aug_paths.append(path)

            if not success:
                continue

            # Build augmented edge multiset
            aug_edges = list(edge_set)
            for path in aug_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    aug_edges.append(key)

            circuit = _find_eulerian_circuit_from_multiset(aug_edges)
            if not circuit:
                continue

            # Cut at augmentation edges
            aug_counter = defaultdict(int)
            for path in aug_paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    aug_counter[key] += 1

            paths = []
            current = [circuit[0]]
            edge_usage = defaultdict(int)

            for i in range(1, len(circuit)):
                u, v = circuit[i - 1], circuit[i]
                key = (u, v) if u <= v else (v, u)
                edge_usage[key] += 1
                current.append(v)

                if aug_counter.get(key, 0) > 0:
                    orig_count = 1 if key in edge_set else 0
                    if edge_usage[key] > orig_count:
                        if len(current) >= 2:
                            paths.append(current)
                        current = [v]
                        aug_counter[key] -= 1

            if len(current) >= 2:
                paths.append(current)

            paths = [p for p in paths if len(p) >= 2]

            if validate(paths, edge_set):
                if best_result is None or len(paths) < len(best_result):
                    best_result = paths
                    if len(paths) == len(odd_vertices) // 2:
                        break

        return best_result

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - _start
        remaining = _DEADLINE - elapsed

        if remaining < 5:
            edges = instance(*spec)
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u <= v else (v, u))
            adj = build_adjacency(edges)
            n = max(max(e) for e in edge_set) + 1 if edge_set else 0
            rng = random.Random(42)
            results.append(_greedy_path_cover_vertex_start(edge_set, adj, n, rng))
            continue

        edges = instance(*spec)
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        adj = build_adjacency(edges)
        n = max(max(e) for e in edge_set) + 1 if edge_set else 0

        # Try multiple strategies and keep the best
        best_paths = None
        best_count = float('inf')

        # Strategy 1: Eulerian augmentation (structural)
        if remaining > 15:
            try:
                aug_paths = _eulerian_augmentation(edge_set, adj, n, time_budget=min(15.0, remaining * 0.3))
                if aug_paths and validate(aug_paths, edge_set):
                    if len(aug_paths) < best_count:
                        best_paths = aug_paths
                        best_count = len(aug_paths)
            except Exception:
                pass

        # Strategy 2: Multiple greedy restarts with vertex-start
        greedy_time = max(2.0, min(20.0, remaining * 0.4))
        start_time = time.time()
        for attempt in range(50):
            if time.time() - start_time > greedy_time or time.time() > _DEADLINE - 5:
                break
            rng = random.Random(attempt)
            paths = _greedy_path_cover_vertex_start(edge_set, adj, n, rng)
            if paths and validate(paths, edge_set):
                if len(paths) < best_count:
                    best_paths = paths
                    best_count = len(paths)

        # Strategy 3: Local search on best found
        if best_paths and remaining > 10:
            ls_time = max(1.0, min(10.0, (remaining - 5) * 0.3))
            try:
                improved = _local_search_merge(best_paths, edge_set, adj, ls_time)
                if validate(improved, edge_set) and len(improved) < best_count:
                    best_paths = improved
                    best_count = len(improved)
            except Exception:
                pass

        # Final fallback
        if not best_paths or not validate(best_paths, edge_set):
            rng = random.Random(0)
            best_paths = _greedy_path_cover_vertex_start(edge_set, adj, n, rng)
            if not validate(best_paths, edge_set):
                best_paths = [[u, v] for (u, v) in sorted(edge_set)]

        results.append(best_paths)

    return results
# EVOLVE-BLOCK-END
