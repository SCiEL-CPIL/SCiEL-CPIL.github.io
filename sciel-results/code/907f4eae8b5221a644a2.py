import time
import random
from collections import defaultdict
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def decompose(edges, time_limit):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Find connected components
        visited = set()
        components = []
        for node in range(max(max(u, v) for u, v in edges) + 1):
            if node in visited or node not in adj:
                continue
            comp = set()
            stack = [node]
            visited.add(node)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            components.append(comp)

        # For each component, find odd-degree vertices
        # and compute minimum T-join (minimum edges to remove to make all degrees even)
        removed_edges = set()

        for comp in components:
            if len(comp) < 3:
                # Can't form a cycle with < 3 vertices
                for u in comp:
                    for v in adj[u]:
                        if v in comp:
                            removed_edges.add((min(u, v), max(u, v)))
                continue

            # Find odd-degree vertices within this component
            odd_vertices = []
            for u in comp:
                deg = sum(1 for v in adj[u] if v in comp)
                if deg % 2 == 1:
                    odd_vertices.append(u)

            if not odd_vertices:
                # Already all even degrees, no edges to remove
                continue

            # Compute shortest path distances between all pairs of odd vertices
            k = len(odd_vertices)
            if k == 0:
                continue

            # dist[i][j] = shortest path distance between odd_vertices[i] and odd_vertices[j]
            dist = [[float('inf')] * k for _ in range(k)]
            parent = [[None] * k for _ in range(k)]

            for i, src in enumerate(odd_vertices):
                bfs_dist = {src: 0}
                bfs_parent = {src: None}
                queue = [src]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    for v in adj[u]:
                        if v in comp and v not in bfs_dist:
                            bfs_dist[v] = bfs_dist[u] + 1
                            bfs_parent[v] = u
                            queue.append(v)

                for j, dst in enumerate(odd_vertices):
                    if j == i:
                        continue
                    if dst in bfs_dist:
                        dist[i][j] = bfs_dist[dst]
                        path = []
                        cur = dst
                        while cur != src:
                            path.append(cur)
                            cur = bfs_parent[cur]
                        path.append(src)
                        path.reverse()
                        parent[i][j] = path

            # Find minimum-weight perfect matching on k odd vertices
            if k > 20:
                matched = [False] * k
                match_pairs = []
                while True:
                    best_i, best_j, best_d = -1, -1, float('inf')
                    for i in range(k):
                        if matched[i]:
                            continue
                        for j in range(i + 1, k):
                            if matched[j]:
                                continue
                            if dist[i][j] < best_d:
                                best_d = dist[i][j]
                                best_i, best_j = i, j
                    if best_i == -1:
                        break
                    matched[best_i] = matched[best_j] = True
                    match_pairs.append((best_i, best_j))
            else:
                size = 1 << k
                dp = [float('inf')] * size
                dp[0] = 0
                match_choice = [None] * size

                for mask in range(size):
                    if dp[mask] == float('inf'):
                        continue
                    first = -1
                    for i in range(k):
                        if not (mask >> i) & 1:
                            first = i
                            break
                    if first == -1:
                        continue
                    for j in range(first + 1, k):
                        if not (mask >> j) & 1:
                            new_mask = mask | (1 << first) | (1 << j)
                            new_cost = dp[mask] + dist[first][j]
                            if new_cost < dp[new_mask]:
                                dp[new_mask] = new_cost
                                match_choice[new_mask] = (first, j)

                match_pairs = []
                mask = (1 << k) - 1
                while mask:
                    i, j = match_choice[mask]
                    match_pairs.append((i, j))
                    mask ^= (1 << i)
                    mask ^= (1 << j)

            # Remove edges along the paths for each matched pair
            for i, j in match_pairs:
                if parent[i][j]:
                    path = parent[i][j]
                    for idx in range(len(path) - 1):
                        u, v = path[idx], path[idx + 1]
                        removed_edges.add((min(u, v), max(u, v)))
                elif dist[i][j] == float('inf'):
                    if adj[odd_vertices[i]]:
                        v = next(iter(adj[odd_vertices[i]]))
                        removed_edges.add((min(odd_vertices[i], v), max(odd_vertices[i], v)))

        # Remaining edges form a subgraph with all even degrees
        remaining = set((min(u, v), max(u, v)) for u, v in edges) - removed_edges

        # Decompose remaining edges into cycles
        pieces = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        def extract_cycle(start):
            """Extract a cycle starting from start using DFS."""
            visited = {start}
            path = [start]

            def dfs(node):
                for neighbor in sorted(rem_adj[node]):
                    if neighbor == start and len(path) >= 3:
                        return path[:]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        result = dfs(neighbor)
                        if result:
                            return result
                        path.pop()
                        visited.discard(neighbor)
                return None

            return dfs(start)

        while remaining:
            start = None
            for v in list(rem_adj.keys()):
                if len(rem_adj[v]) >= 2:
                    start = v
                    break

            if start is None:
                break

            cycle = extract_cycle(start)
            if cycle is None:
                u, v = next(iter(remaining))
                pieces.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue

            pieces.append(cycle)
            n_c = len(cycle)
            for i in range(n_c):
                u = cycle[i]
                v = cycle[(i + 1) % n_c]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)

        for u, v in sorted(remaining):
            pieces.append([u, v])

        for u, v in sorted(removed_edges):
            pieces.append([u, v])

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
