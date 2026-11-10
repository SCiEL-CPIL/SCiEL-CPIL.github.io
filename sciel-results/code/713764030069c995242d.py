"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Track remaining edges
        remaining = set(edges)
        pieces = []

        def find_cycle(start, max_len=20):
            """Find a cycle starting and ending at start using DFS."""
            if len(remaining) < 3:
                return None
            # Try to find a cycle
            def dfs(node, path, visited):
                if len(path) >= max_len:
                    return None
                for nb in sorted(adj.get(node, [])):
                    if (min(node, nb), max(node, nb)) not in remaining:
                        continue
                    if nb == start and len(path) >= 3:
                        return path[:]
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        result = dfs(nb, path, visited)
                        if result:
                            return result
                        path.pop()
                        visited.discard(nb)
                return None
            return dfs(start, [start], {start})

        # Greedily extract cycles
        while remaining:
            # Find a vertex with degree >= 2 to start a cycle search
            cycle_found = False
            candidates = sorted([v for v in adj if len([e for e in remaining if v in e]) >= 2])

            for start in candidates:
                cycle = find_cycle(start)
                if cycle:
                    # Extract this cycle
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        edge = (min(u, v), max(u, v))
                        remaining.discard(edge)
                    pieces.append(cycle[:])
                    cycle_found = True
                    break

            if not cycle_found:
                # No more cycles; take remaining edges as singles
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
                break

        # Now try to merge: replace some single edges with cycles
        # and try to find longer cycles to reduce piece count
        # Try to find cycles among the single edges combined with
        # breaking some existing cycles

        # Better approach: try to find cycles in the remaining edge set
        # by looking for 3-cycles, 4-cycles, etc.
        # Actually, let's try a different strategy:
        # 1. First, find all possible cycles (3-cycles, 4-cycles, etc.)
        # 2. Use a greedy set cover approach

        # Let me redo this more carefully
        # Strategy: repeatedly find the shortest available cycle, remove it,
        # until no cycles remain, then the rest are single edges.

        # Actually the above greedy approach already does this. Let me verify
        # it's correct and try to improve it.

        # Improvement: try to find cycles of various lengths, prefer shorter
        # ones that use edges that are "harder" to form cycles with

        # Let me try a different approach: use a maximum matching-like strategy
        # to find edge-disjoint cycles

        # For now, return what we have
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
