import random
import time
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

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

    def bfs_distances(start, adj):
        """BFS from start, return dict of distances."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    def find_path_dfs(start, end, remaining_edges, adj, max_depth=80):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]

        # Quick check: is there a path in the remaining graph?
        # Use BFS on remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Check connectivity
        if end not in rem_adj[start] and start != end:
            # Need to verify via BFS
            visited = {start}
            queue = deque([start])
            found = False
            while queue:
                node = queue.popleft()
                if node == end:
                    found = True
                    break
                for nb in rem_adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if not found:
                return None

        best_path = None

        def dfs(node, path, visited):
            nonlocal best_path
            if node == end:
                if best_path is None or len(path) < len(best_path):
                    best_path = list(path)
                return True
            if len(path) > max_depth:
                return False

            neighbors = list(rem_adj[node])
            # Heuristic: prefer neighbors that are closer to end (in original graph)
            # and have higher remaining degree
            def neighbor_score(nb):
                if nb in visited:
                    return (1, 0, 0)
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    return (1, 0, 0)
                # Prefer neighbors with more remaining edges (more flexible)
                deg = len(rem_adj[nb])
                # Prefer neighbors closer to end
                return (0, -deg, 0)

            neighbors.sort(key=neighbor_score)

            for nb in neighbors:
                if nb in visited:
                    continue
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue

                remaining_edges.discard(e)
                visited.add(nb)
                path.append(nb)

                dfs(nb, path, visited)

                path.pop()
                visited.discard(nb)
                remaining_edges.add(e)

                if best_path is not None:
                    return True

            return False

        dfs(start, [start], {start})
        return best_path

    def find_path_greedy(start, end, remaining_edges, adj):
        """Greedy path finding as fallback."""
        if start == end:
            return [start]

        path = [start]
        visited = {start}
        current = start

        while current != end and len(path) < 100:
            candidates = []
            for nb in adj[current]:
                if nb in visited:
                    continue
                e = tuple(sorted((current, nb)))
                if e in remaining_edges:
                    # Score: prefer higher degree neighbors
                    deg = sum(1 for nb2 in adj[nb] if tuple(sorted((nb, nb2))) in remaining_edges)
                    candidates.append((deg, nb))

            if not candidates:
                return None

            # Pick highest degree
            candidates.sort(reverse=True)
            next_node = candidates[0][1]
            e = tuple(sorted((current, next_node)))
            remaining_edges.discard(e)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        if current != end:
            # Roll back
            for i in range(len(path) - 1, 0, -1):
                e = tuple(sorted((path[i-1], path[i])))
                remaining_edges.add(e)
            return None

        return path

    def decompose(edges):
        if not edges:
            return []

        adj = build_adj(edges)
        remaining_edges = set(edges)
        paths = []

        # Compute degrees
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1

        # Find odd-degree vertices
        odd_vertices = sorted([v for v in degree if degree[v] % 2 == 1])

        if len(odd_vertices) >= 2:
            m = len(odd_vertices)

            # Compute all-pairs BFS distances from each odd vertex
            all_dists = []
            for ov in odd_vertices:
                all_dists.append(bfs_distances(ov, adj))

            # Build distance matrix
            dist_matrix = np.zeros((m, m))
            for i in range(m):
                for j in range(i + 1, m):
                    d = all_dists[i].get(odd_vertices[j], 1000)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            # Find minimum weight perfect matching
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            # Extract pairs
            pairs = []
            seen = set()
            for r, c in zip(row_ind, col_ind):
                if r != c:
                    key = (min(odd_vertices[r], odd_vertices[c]), max(odd_vertices[r], odd_vertices[c]))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((odd_vertices[r], odd_vertices[c]))

            # For each pair, try to find a path
            # Try multiple orderings to maximize success
            for attempt in range(3):
                if time.time() - start_time > time_limit - 5:
                    break

                if attempt == 0:
                    # Original order
                    ordered_pairs = pairs
                elif attempt == 1:
                    # Reverse
                    ordered_pairs = pairs[::-1]
                else:
                    # Sort by distance (shortest first)
                    ordered_pairs = sorted(pairs, key=lambda p: all_dists[odd_vertices.index(p[0])].get(p[1], 1000))

                found_all = True
                for (a, b) in ordered_pairs:
                    if time.time() - start_time > time_limit - 3:
                        found_all = False
                        break

                    # Check if a and b still have odd degree (i.e., still need to be endpoints)
                    # Actually, we should check if they're still in the remaining graph
                    if not remaining_edges:
                        break

                    # Try DFS first
                    path = find_path_dfs(a, b, remaining_edges, adj, max_depth=60)
                    if path is None:
                        # Try greedy
                        path = find_path_greedy(a, b, remaining_edges, adj)

                    if path and len(path) >= 2:
                        paths.append(path)
                        # Update degrees
                        for i in range(len(path) - 1):
                            degree[path[i]] -= 1
                            degree[path[i + 1]] -= 1
                    else:
                        found_all = False
                        break

                if found_all:
                    break

        # Greedy: cover remaining edges
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                # Emergency: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break

            # Recompute degrees for remaining edges
            rem_degree = defaultdict(int)
            for u, v in remaining_edges:
                rem_degree[u] += 1
                rem_degree[v] += 1

            # Pick an edge to start a path
            # Choose edge where sum of endpoint degrees is highest
            best_edge = None
            best_key = -1
            for e in remaining_edges:
                u, v = e
                k = rem_degree[u] + rem_degree[v]
                if k > best_key:
                    best_key = k
                    best_edge = e

            if best_edge is None:
                break

            u, v = best_edge
            path = [u, v]
            remaining_edges.discard(best_edge)
            path_set = set(path)

            # Extend from u end
            while True:
                current = path[0]
                candidates = [nb for nb in adj[current]
                             if tuple(sorted((current, nb))) in remaining_edges
                             and nb not in path_set]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.insert(0, next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1

            # Extend from v end
            while True:
                current = path[-1]
                candidates = [nb for nb in adj[current]
                             if tuple(sorted((current, nb))) in remaining_edges
                             and nb not in path_set]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -rem_degree[nb])
                next_node = candidates[0]
                path.append(next_node)
                path_set.add(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
                rem_degree[current] -= 1
                rem_degree[next_node] -= 1

            paths.append(path)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
