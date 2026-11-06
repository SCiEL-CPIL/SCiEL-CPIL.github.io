"""
Path decomposition of random connected graphs using Chinese Postman + Euler path
extraction with heuristics to minimize path count.
"""

def construct():
    import random as _random
    from collections import defaultdict

    def instance_graph(n, p, seed):
        rng = _random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def decompose(edges):
        """Decompose graph into minimum number of paths (greedy heuristic)."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Count degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1

        # Find odd-degree vertices
        odd = [v for v in deg if deg[v] % 2 == 1]

        # If no odd vertices, we have an Eulerian circuit -> 1 path
        # If 2k odd vertices, we need at least k paths

        # Strategy: pair up odd-degree vertices and find paths between them
        # to make all degrees even, then find Eulerian circuit.
        # But that gives us 1 circuit. Instead, we want to minimize path count.

        # Better strategy: Use the fact that a graph with 2k odd-degree vertices
        # can be decomposed into k paths. We try to find k paths that cover all edges.

        # Greedy approach: repeatedly find longest possible simple paths,
        # removing edges as we go, until no edges remain.

        # Edge set for tracking
        remaining = set()
        for e in edges:
            remaining.add(e)

        # Helper to get neighbors via remaining edges
        def get_adj_remaining(node, rem):
            neighbors = set()
            for e in rem:
                if e[0] == node:
                    neighbors.add(e[1])
                elif e[1] == node:
                    neighbors.add(e[0])
            return neighbors

        paths = []
        rem = set(remaining)

        # Find start vertices: prefer odd-degree vertices in original graph
        # Sort by degree to start from "ends"
        start_candidates = sorted(deg.keys(), key=lambda x: (-deg[x], x))

        # For better path coverage, start from odd-degree vertices first
        odd_set = set(odd)

        while rem:
            # Find a good starting vertex: one with odd degree in remaining graph,
            # or a vertex with degree > 0
            rem_deg = defaultdict(int)
            for u, v in rem:
                rem_deg[u] += 1
                rem_deg[v] += 1

            # Prefer vertices with odd remaining degree
            odd_rem = [v for v in rem_deg if rem_deg[v] % 2 == 1]
            if odd_rem:
                start = odd_rem[0]
            else:
                # All even, pick any vertex with degree > 0
                start = list(rem_deg.keys())[0] if rem_deg else None
                if start is None:
                    break

            # Walk greedily to build a long path
            path = [start]
            visited_edges_in_path = set()
            current = start

            while True:
                # Find neighbors via remaining edges
                neighbors = []
                for e in rem:
                    if current in e:
                        other = e[0] if e[1] == current else e[1]
                        if e not in visited_edges_in_path:
                            neighbors.append(other)

                if not neighbors:
                    break

                # Choose next vertex: prefer one that leads to more edges (avoid dead ends early)
                # Simple heuristic: pick neighbor with highest remaining degree
                def neighbor_degree(nbr):
                    cnt = 0
                    for e in rem:
                        if e in visited_edges_in_path:
                            continue
                        if nbr in e:
                            cnt += 1
                    return cnt

                # Sort by degree descending, then by vertex id for determinism
                neighbors.sort(key=lambda x: (-neighbor_degree(x), x))
                next_v = neighbors[0]

                edge = (min(current, next_v), max(current, next_v))
                visited_edges_in_path.add(edge)
                rem.discard(edge)
                path.append(next_v)
                current = next_v

            if len(path) > 1:
                paths.append(path)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance_graph(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
