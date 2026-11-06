"""Decomposition of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
from collections import defaultdict, deque

def construct():
    """
    Decompose each instance graph into edge-disjoint cycles and single edges,
    minimizing the total number of pieces.
    """
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into minimum pieces (cycles + single edges)."""
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(tuple(sorted(e)) for e in edges)
        cycles = []

        # Repeatedly find a shortest cycle in the remaining graph
        while remaining:
            # Find shortest cycle using BFS from each vertex
            best_cycle = None
            best_len = float('inf')

            vertices = sorted(set(v for e in remaining for v in e))
            for start in vertices:
                # BFS to find shortest cycle through start
                # dist[v] = shortest path length from start to v
                # parent[v] = previous vertex on shortest path
                dist = {start: 0}
                parent = {start: None}
                queue = deque([start])

                while queue:
                    u = queue.popleft()
                    for v in sorted(adj[u]):
                        edge = tuple(sorted((u, v)))
                        if edge not in remaining:
                            continue
                        if v == start and dist[u] + 1 >= 3:
                            # Found a cycle
                            # Reconstruct path
                            path = [u, start]
                            cur = u
                            while cur != start:
                                cur = parent[cur]
                                if cur is not None:
                                    path.append(cur)
                            path.reverse()
                            # path goes from start to ... to u, then u-start
                            # Actually: start -> ... -> u -> start
                            # Let me reconstruct properly
                            path = [start]
                            cur = u
                            while cur != start:
                                path.append(cur)
                                cur = parent[cur]
                            # path is [start, ..., u], cycle is path + [start]
                            cycle = path  # closing edge implied
                            if len(cycle) < best_len:
                                best_cycle = cycle
                                best_len = len(cycle)
                                if best_len == 3:
                                    break
                        elif v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)

                if best_len == 3:
                    break

            if best_cycle is None:
                break

            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    remaining.remove(edge)

            cycles.append(list(best_cycle))

        # Build pieces
        pieces = [list(c) for c in cycles]
        for edge in remaining:
            pieces.append(list(edge))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
