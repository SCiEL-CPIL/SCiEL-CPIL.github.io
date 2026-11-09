"""Decompose three fixed random graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n):
        if not edges:
            return []
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        remaining = set(edges)
        pieces = []

        def try_find_cycle(start, min_len=3, max_len=None, deadline=None):
            """DFS to find a cycle through start. Returns list of vertices or None."""
            if max_len is None:
                max_len = n
            visited = set()
            stack = [(start, [start])]
            while stack:
                if deadline and time.time() > deadline:
                    return None
                node, path = stack.pop()
                visited.add(node)
                for nb in adj[node]:
                    if nb == start and len(path) >= min_len:
                        return path
                    if nb not in visited and len(path) < max_len:
                        stack.append((nb, path + [nb]))
            return None

        def find_longest_cycle(start, deadline=None):
            """Try to find a long cycle starting from start."""
            # Try progressively longer cycles
            best = None
            for target_len in range(n, 2, -1):
                if deadline and time.time() > deadline:
                    break
                # Use a simple backtracking for a cycle of exactly target_len
                # This is expensive, so limit to smaller targets
                if target_len > 15:
                    continue
                # Simple DFS for cycle of length target_len
                visited = {start}
                stack = [(start, [start])]
                found = None
                while stack:
                    if deadline and time.time() > deadline:
                        return best
                    node, path = stack.pop()
                    for nb in adj[node]:
                        if nb == start and len(path) == target_len - 1:
                            found = path
                            break
                        if nb not in visited and len(path) < target_len:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                            visited.discard(nb)
                    if found:
                        break
                if found:
                    best = found
                    break
            return best

        # Greedy: repeatedly find cycles in the remaining graph
        # Start with vertices of highest degree
        while remaining and (not deadline or time.time() < deadline):
            if deadline and time.time() > deadline - 2:
                break

            # Find a vertex with max degree in remaining graph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            if not deg:
                break

            # Pick vertex with max degree
            start = max(deg, key=lambda x: deg[x])

            # Try to find a cycle through this vertex
            cycle = None

            # First try a simple DFS-based cycle finder
            # We'll do a proper DFS that tracks the path
            visited = set()
            parent = {}
            found_cycle = False

            def dfs_cycle(v, path):
                nonlocal found_cycle, cycle
                if found_cycle:
                    return
                visited.add(v)
                for nb in adj[v]:
                    if nb == path[0] and len(path) >= 3:
                        cycle = list(path)
                        found_cycle = True
                        return
                    if nb not in visited:
                        dfs_cycle(nb, path + [nb])
                        if found_cycle:
                            return
                    # Check if nb is in path but not the first (for longer cycles)
                    # Actually, for a simple cycle through start, we just need to return to start
                visited.discard(v)

            dfs_cycle(start, [start])
            if cycle is None:
                # Try starting from other high-degree vertices
                for v in sorted(deg, key=lambda x: -deg[x])[:5]:
                    if found_cycle:
                        break
                    visited = set()
                    cycle = None
                    dfs_cycle(v, [v])

            if cycle and len(cycle) >= 3:
                # Extract edges of this cycle
                cycle_edges = set()
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        cycle_edges.add(e)
                    else:
                        cycle = None
                        break
                if cycle:
                    pieces.append(cycle)
                    remaining -= cycle_edges
                else:
                    # Couldn't form a valid cycle, try shorter
                    pass
            else:
                # No cycle found, add as edge
                # Pick an edge incident to start
                edge = None
                for u, v in remaining:
                    if u == start or v == start:
                        edge = (u, v)
                        break
                if edge is None:
                    edge = next(iter(remaining))
                pieces.append(list(edge))
                remaining.discard(edge)

        # Add remaining edges
        for e in remaining:
            pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE - 5:
            # Not enough time, just return edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose(edges, n)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
