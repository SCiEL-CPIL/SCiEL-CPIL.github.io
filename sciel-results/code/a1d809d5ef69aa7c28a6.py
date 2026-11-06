"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
import time
import random
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: evaluator kills after 150s; keep a safety margin.
    deadline = time.time() + 130  # 20s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_decompose(edges, adj, max_time=None):
        """
        Decompose edges into cycles and single edges using a greedy approach.
        Try to find long cycles repeatedly.
        """
        remaining_edges = set(edges)
        remaining_adj = defaultdict(set)
        for u, v in remaining_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)

        pieces = []

        def find_cycle(start, remaining_adj, remaining_edges):
            """
            Try to find a cycle starting from 'start' using DFS.
            Returns a list of vertices forming the cycle, or None.
            """
            # We want a cycle that includes 'start'.
            # Do a DFS trying to return to 'start' without repeating vertices.
            # Use backtracking with a depth limit for speed.
            max_depth = 50  # limit to avoid exponential blowup
            visited = set()
            path = []

            def dfs(node, depth):
                if depth > max_depth:
                    return None
                if node == start and len(path) >= 3:
                    return list(path)
                visited.add(node)
                path.append(node)
                # Sort neighbors for deterministic behavior
                for nb in sorted(remaining_adj.get(node, [])):
                    if nb == start and len(path) >= 3:
                        result = list(path)
                        path.pop()
                        visited.discard(node)
                        return result
                    if nb not in visited:
                        result = dfs(nb, depth + 1)
                        if result is not None:
                            path.pop()
                            visited.discard(node)
                            return result
                path.pop()
                visited.discard(node)
                return None

            return dfs(start, 0)

        # Repeatedly find cycles
        while remaining_edges:
            if max_time and time.time() > max_time:
                break

            # Pick a random edge to start from
            edge = random.choice(list(remaining_edges))
            start = edge[0]

            # Try to find a cycle from start
            cycle = find_cycle(start, remaining_adj, remaining_edges)

            if cycle and len(cycle) >= 3:
                # Remove the edges in the cycle
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        remaining_adj[u].discard(v)
                        remaining_adj[v].discard(u)
                pieces.append(cycle)
            else:
                # Remove one edge as a single piece
                u, v = edge
                remaining_edges.remove(edge)
                remaining_adj[u].discard(v)
                remaining_adj[v].discard(u)
                pieces.append([u, v])

        # Any remaining edges become single-edge pieces
        for e in remaining_edges:
            pieces.append(list(e))

        return pieces

    def multi_start_decompose(edges, adj, n_starts=500, max_time=None):
        """
        Try multiple random starts and keep the best decomposition.
        """
        best_pieces = None
        best_count = float('inf')

        for i in range(n_starts):
            if max_time and time.time() > max_time:
                break
            pieces = greedy_decompose(edges, adj, max_time=max_time)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces

        return best_pieces if best_pieces is not None else greedy_decompose(edges, adj)

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for n, p, seed in specs:
        edges = instance(n, p, seed)
        adj = build_adjacency(edges)

        # Use multi-start greedy decomposition
        pieces = multi_start_decompose(edges, adj, n_starts=200, max_time=deadline)

        # If we ran out of time, fall back to a simple decomposition
        if pieces is None or time.time() > deadline - 5:
            pieces = [[list(e)] for e in edges]

        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
