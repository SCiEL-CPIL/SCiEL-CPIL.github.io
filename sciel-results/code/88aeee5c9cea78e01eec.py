"""Decompose graphs into cycles and edges, minimizing piece count."""
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin under 150s

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, used_vertices, max_depth=30):
        """Find a vertex-disjoint cycle avoiding used_vertices. Returns list of vertices or None."""
        n = len(adj)
        available = [v for v in range(n) if v not in used_vertices]
        if len(available) < 3:
            return None

        best_cycle = None

        # Try starting from each available vertex
        for start in available:
            if start in used_vertices:
                continue
            # DFS to find longest cycle through start
            # Path: start -> ... -> start
            stack = [(start, [start], set([start]))]
            while stack:
                if time.time() > deadline:
                    break
                node, path, visited = stack.pop()
                if len(path) >= 3 and start in adj[node]:
                    # Found a cycle
                    cycle = path[:]
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    continue
                # Extend
                for nb in adj[node]:
                    if nb in visited:
                        if nb == start and len(path) >= 3:
                            cycle = path[:]
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                        continue
                    if nb in used_vertices:
                        continue
                    new_path = path + [nb]
                    new_visited = visited | {nb}
                    # Prune: if we already have a cycle of this length or longer, skip
                    if best_cycle is not None and len(new_path) + 1 <= len(best_cycle):
                        # Even extending by 1 won't beat best (need to close)
                        # Actually we need at least 1 more edge to close, so len(new_path)+1 is the cycle length
                        # If that's <= best, skip
                        continue
                    stack.append((nb, new_path, new_visited))

        return best_cycle

    def decompose(n, edges):
        """Decompose graph into cycles and single edges."""
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))

        # Sort adjacency lists for deterministic behavior
        for i in range(n):
            adj[i].sort()

        pieces = []
        used_vertices = set()
        remaining_edges = set(edge_set)

        # Iteratively find longest vertex-disjoint cycles
        while True:
            if time.time() > deadline:
                break
            cycle = find_longest_cycle(adj, used_vertices)
            if cycle is None:
                break
            # Verify all edges in cycle exist
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e not in remaining_edges:
                    valid = False
                    break
            if not valid:
                # This shouldn't happen, but handle gracefully
                break

            # Add cycle as a piece
            pieces.append(cycle[:])
            # Mark vertices as used
            for v in cycle:
                used_vertices.add(v)
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                remaining_edges.discard(e)

        # Remaining edges become single-edge pieces
        for u, v in sorted(remaining_edges):
            pieces.append([u, v])

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)

    return results
