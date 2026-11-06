"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    """Decompose three fixed graphs into few cycles and edges."""
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n):
        edge_set = set(tuple(e) for e in edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []

        def find_cycle(adj, edge_set):
            """Find a cycle in the current graph. Returns list of vertices or None."""
            # Get vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None

            random.shuffle(candidates)
            for start in candidates:
                # DFS from start
                stack = [(start, None, [start])]
                visited = {start}

                while stack:
                    node, parent, path = stack.pop()

                    neighbors = list(adj[node])
                    random.shuffle(neighbors)

                    for nb in neighbors:
                        edge = tuple(sorted((node, nb)))
                        if edge not in edge_set:
                            continue

                        if nb == start and len(path) >= 3:
                            return path[:]

                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))

            return None

        def remove_cycle(cycle, edge_set, adj):
            m = len(cycle)
            for i in range(m):
                u, v = cycle[i], cycle[(i + 1) % m]
                edge = tuple(sorted((u, v)))
                if edge in edge_set:
                    edge_set.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)

        # Greedy: extract cycles
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            if not edge_set:
                break

            cycle = find_cycle(adj, edge_set)
            if cycle is None:
                break

            # Try to find a longer cycle with a few more attempts
            best = cycle
            for _ in range(3):
                alt = find_cycle(adj, edge_set)
                if alt and len(alt) > len(best):
                    best = alt

            remove_cycle(best, edge_set, adj)
            pieces.append(best)

        # Remaining edges
        for u, v in sorted(edge_set):
            pieces.append([u, v])

        return pieces

    results = []
    for n, p, seed in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - start_time > DEADLINE - 5:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
