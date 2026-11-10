import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_odd_degree_paths(edges, adj, deadline):
        """
        Find edge-disjoint paths covering all edges using the odd-degree vertex structure.
        The minimum number of paths is at least ceil(#odd/2).
        We find a matching on odd-degree vertices and trace paths between matched pairs.
        """
        # Compute degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1

        # Find odd-degree vertices
        odd_vertices = [v for v in deg if deg[v] % 2 == 1]

        if not odd_vertices:
            # Eulerian: one closed walk covers all edges, but we need paths
            # Find an Eulerian circuit and split into paths
            # Actually, if all degrees are even, we can have 0 or 1 paths (a cycle)
            # But we need simple paths, so we may need to split
            # For now, use greedy
            return None

        n_odd = len(odd_vertices)
        target_paths = (n_odd + 1) // 2  # ceil(n_odd / 2)

        # We need to find a perfect matching on odd_vertices
        # such that for each matched pair (u, v), there exists a simple path
        # using only remaining edges that connects u to v.
        # Then we decompose the remaining edges into these paths.

        # Strategy: 
        # 1. Find a minimum-weight perfect matching on odd vertices
        #    where weight(u,v) = shortest path distance in the graph
        # 2. For each matched pair, find a simple path
        # 3. Remove those edges and repeat for remaining odd vertices

        # Actually, a cleaner approach:
        # The graph can be decomposed into edge-disjoint paths iff we can
        # pair up the odd-degree vertices such that each pair is connected
        # by a path in the graph.
        
        # Let's use a different approach:
        # Start with all edges. Repeatedly find a path between two odd-degree
        # vertices, remove its edges, and update degrees.
        
        remaining = set(map(tuple, edges))
        cur_deg = defaultdict(int)
        for e in remaining:
            cur_deg[e[0]] += 1
            cur_deg[e[1]] += 1

        paths = []

        # Build adjacency from remaining edges for quick lookup
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        while remaining:
            if time.time() > deadline:
                # Fallback
                for e in remaining:
                    paths.append(list(e))
                return paths

            # Find current odd-degree vertices
            odd_verts = [v for v in cur_deg if cur_deg[v] % 2 == 1 and cur_deg[v] > 0]

            if not odd_verts:
                # All even: find any path (or cycle)
                # Just pick any edge and extend
                any_edge = next(iter(remaining))
                start_v = any_edge[0]
            else:
                # Pick a pair of odd vertices to connect
                # Use greedy: pick the pair with shortest path
                if len(odd_verts) >= 2:
                    # Try to find shortest path between pairs
                    best_pair = None
                    best_dist = float('inf')
                    # Limit search for speed
                    sample = odd_verts[:min(10, len(odd_verts))]
                    for i in range(len(sample)):
                        for j in range(i+1, len(sample)):
                            u, v = sample[i], sample[j]
                            # BFS to find shortest path
                            dist = _bfs_shortest_path(u, v, rem_adj)
                            if dist is not None and dist < best_dist:
                                best_dist = dist
                                best_pair = (u, v)
                    
                    if best_pair is None:
                        # Just pick first two
                        best_pair = (odd_verts[0], odd_verts[1])
                else:
                    best_pair = (odd_verts[0], odd_verts[0])  # degenerate

                start_v = best_pair[0]
                end_v = best_pair[1]

            # Find a simple path from start_v to end_v using remaining edges
            # Prefer paths that go through high-degree vertices to leave
            # low-degree vertices for later
            path = _find_path_heuristic(start_v, end_v, rem_adj, remaining, cur_deg, deadline)

            if path is None or len(path) < 2:
                # Fallback: just take one edge
                e = next(iter(remaining))
                path = list(e)

            # Remove path edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                remaining.discard(edge)
                cur_deg[u] -= 1
                cur_deg[v] -= 1
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)

            paths.append(path)

        return paths

    def _bfs_shortest_path(start, end, adj):
        """BFS to find shortest path length (number of edges) from start to end."""
        if start == end:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            v, d = queue.pop(0)
            for u in adj[v]:
                if u not in visited:
                    if u == end:
                        return d + 1
                    visited.add(u)
                    queue.append((u, d + 1))
        return None

    def _find_path_heuristic(start, end, adj, remaining, deg, deadline):
        """
        Find a simple path from start to end using only edges in remaining.
        Uses a heuristic to prefer going through high-degree vertices.
        """
        if start == end:
            return [start]

        # DFS with heuristic
        path = [start]
        visited = {start}

        def dfs(v):
            if v == end:
                return True
            # Sort neighbors by degree (prefer high-degree to keep options open)
            neighbors = sorted(adj[v], key=lambda u: -deg.get(u, 0))
            for u in neighbors:
                if u not in visited:
                    edge = (min(v, u), max(v, u))
                    if edge in remaining:
                        visited.add(u)
                        path.append(u)
                        if dfs(u):
                            return True
                        path.pop()
                        visited.discard(u)
            return False

        if dfs(start):
            return path
        return None

    def greedy_walk(adj, remaining_edges, deg, start_v, rng, temperature, avoid_cycles, target_end=None):
        """
        Perform a single greedy walk from start_v.
        If target_end is given, try to reach it.
        """
        path = [start_v]
        visited = {start_v} if avoid_cycles else set()

        while True:
            if time.time() > deadline:
                break
            v = path[-1]
            valid_neighbors = []
            for u in adj[v]:
                edge = (min(v, u), max(v, u))
                if edge in remaining_edges:
                    if avoid_cycles and u in visited:
                        continue
                    valid_neighbors.append(u)

            if not valid_neighbors:
                break

            if target_end is not None and target_end in valid_neighbors:
                next_v = target_end
            elif temperature <= 0.01:
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                weights = []
                for u in valid_neighbors:
                    d = deg.get(u, 0)
                    weights.append(1.0 / (d + 1.0))
                total_w = sum(weights)
                if total_w == 0:
                    next_v = valid_neighbors[0]
                else:
                    r = rng.random() * total_w
                    cum = 0
                    next_v = valid_neighbors[-1]
                    for i, u in enumerate(valid_neighbors):
                        cum += weights[i]
                        if r <= cum:
                            next_v = u
                            break

            edge = (min(v, next_v), max(v, next_v))
            remaining_edges.discard(edge)
            deg[v] -= 1
            deg[next_v] -= 1
            path.append(next_v)
            if avoid_cycles:
                visited.add(next_v)

            if target_end is not None and next_v == target_end:
                break

        return path

    def path_cover_single(edges, adj, rng, temperature, start_mode, deadline):
        """
        Run one full path-cover attempt with given parameters.
        Returns list of paths.
        """
        remaining = set(map(tuple, edges))
        deg = defaultdict(int)
        for e in remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1

        paths = []
        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    paths.append(list(e))
                break

            # Choose start vertex
            if start_mode == 'leaf':
                leaves = [v for e in remaining for v in e if deg[v] == 1]
                if leaves:
                    start_v = rng.choice(leaves) if temperature > 0.01 else leaves[0]
                else:
                    min_deg = min(deg[v] for e in remaining for v in e)
                    candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                    start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            elif start_mode == 'min_deg':
                min_deg = min(deg[v] for e in remaining for v in e)
                candidates = [v for e in remaining for v in e if deg[v] == min_deg]
                start_v = rng.choice(candidates) if temperature > 0.01 else candidates[0]
            else:
                all_vertices = set()
                for e in remaining:
                    all_vertices.add(e[0])
                    all_vertices.add(e[1])
                start_v = rng.choice(list(all_vertices))

            path = greedy_walk(adj, remaining, deg, start_v, rng, temperature, True)
            if len(path) > 1:
                paths.append(path)
            else:
                for u in adj[start_v]:
                    edge = (min(start_v, u), max(start_v, u))
                    if edge in remaining:
                        path.append(u)
                        remaining.discard(edge)
                        deg[start_v] -= 1
                        deg[u] -= 1
                        break
                if len(path) > 1:
                    paths.append(path)
                else:
                    paths.append(path)

        for e in remaining:
            paths.append(list(e))

        return paths

    def merge_paths(paths, edges, deadline):
        """
        Try to merge paths by finding connecting edges between endpoints.
        """
        if len(paths) < 2:
            return paths

        all_edges = set(map(tuple, edges))
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                used_edges.add((min(u, v), max(u, v)))

        remaining_edges = all_edges - used_edges

        improved = True
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break

            for i in range(len(paths)):
                if improved:
                    break
                for j in range(len(paths)):
                    if i == j:
                        continue

                    end_i = paths[i][-1]
                    start_j = paths[j][0]
                    edge = (min(end_i, start_j), max(end_i, start_j))
                    if edge in remaining_edges:
                        if start_j not in paths[i]:
                            paths[i] = paths[i] + paths[j]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break

                    start_i = paths[i][0]
                    end_j = paths[j][-1]
                    edge = (min(start_i, end_j), max(start_i, end_j))
                    if edge in remaining_edges:
                        if end_j not in paths[i]:
                            paths[i] = list(reversed(paths[j])) + paths[i]
                            remaining_edges.discard(edge)
                            paths.pop(j)
                            improved = True
                            break

        return paths

    def solve_graph(edges, n, deadline):
        """
        Solve path cover for a single graph.
        Returns best list of paths found.
        """
        adj = build_adj(edges)
        best_cover = None
        best_count = len(edges)

        # First try the structural odd-degree approach
        result = find_odd_degree_paths(edges, adj, deadline)
        if result is not None:
            result = merge_paths(result, edges, deadline)
            count = len(result)
            if count < best_count:
                best_count = count
                best_cover = result

        # Try greedy strategies
        strategies = [
            (0.0, 'leaf'),
            (0.0, 'min_deg'),
            (0.2, 'leaf'),
            (0.4, 'leaf'),
            (0.6, 'leaf'),
            (0.8, 'leaf'),
            (1.0, 'leaf'),
            (0.3, 'min_deg'),
            (0.5, 'min_deg'),
            (0.7, 'min_deg'),
            (0.9, 'min_deg'),
            (0.5, 'random'),
            (0.8, 'random'),
        ]

        for temp, mode in strategies:
            if time.time() > deadline:
                break

            for trial in range(30):
                if time.time() > deadline:
                    break

                rng = random.Random(hash((temp, mode, trial)) % (2**32))
                cover = path_cover_single(edges, adj, rng, temp, mode, deadline)
                cover = merge_paths(cover, edges, deadline)
                count = len(cover)

                if count < best_count:
                    best_count = count
                    best_cover = cover

                    if best_count <= 8:
                        break

            if time.time() > deadline:
                break
            if best_count <= 8:
                break

        if best_cover is None:
            best_cover = [list(e) for e in edges]

        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        cover = solve_graph(edges, n, deadline)
        results.append(cover)

    return results
