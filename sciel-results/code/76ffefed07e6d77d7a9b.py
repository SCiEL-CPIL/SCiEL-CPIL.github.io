"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, time_limit=40.0):
    """
    Decompose a graph into edge-disjoint cycles and edges, minimizing total pieces.
    Strategy:
      1. Repeatedly find a simple cycle using fast DFS and extract it.
      2. Prioritize high-degree vertices to find cycles.
      3. After no more cycles can be found, remaining edges become single-edge pieces.
    """
    if not edges:
        return []

    adj = defaultdict(set)
    alive_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        alive_edges.add((u, v))
        alive_edges.add((v, u))

    pieces = []
    start_time = time.time()

    def find_cycle(start):
        """
        Find a simple cycle starting and ending at 'start' using iterative DFS.
        Returns list of vertices in the cycle or None.
        """
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            u, path = stack.pop()
            # Sort neighbors by degree descending to heuristic favor longer cycles
            neighbors = sorted(adj[u], key=lambda x: len(adj[x]), reverse=True)
            for v in neighbors:
                if (u, v) not in alive_edges:
                    continue
                if v == start and len(path) >= 2:
                    # Found a cycle
                    return path[:]
                if v not in visited:
                    visited.add(v)
                    stack.append((v, path + [v]))
        
        return None

    # Main loop: extract cycles
    while True:
        if time.time() - start_time > time_limit:
            break
        if not alive_edges:
            break

        # Find a vertex with degree >= 2, prioritizing high degree
        vertices = sorted([v for v in range(n) if len(adj[v]) >= 2], key=lambda v: len(adj[v]), reverse=True)
        
        if not vertices:
            break

        cycle = None
        for v in vertices:
            if time.time() - start_time > time_limit:
                break
            cycle = find_cycle(v)
            if cycle is not None:
                break
        
        if cycle is None:
            break

        # Remove edges in the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            alive_edges.discard((u, v))
            alive_edges.discard((v, u))
            adj[u].discard(v)
            adj[v].discard(u)

        pieces.append([int(v) for v in cycle])

    # Remaining edges
    remaining = set()
    for u, v in alive_edges:
        if u < v:
            remaining.add((u, v))

    for u, v in sorted(remaining):
        pieces.append([u, v])

    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = 145.0 - elapsed
        graphs_left = len(specs) - i
        alloc_time = remaining_time / graphs_left
        alloc_time = max(2.0, min(alloc_time, 45.0))

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)

    return results
