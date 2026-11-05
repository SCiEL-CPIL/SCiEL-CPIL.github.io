"""Decompose fixed instance graphs into few cycles and edges."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycle_in_subgraph(adj_list, n, rng):
    """Find a cycle in the subgraph. Returns list of vertices or None."""
    # Try to find a cycle using DFS from each vertex
    visited = [False] * n
    parent = [-1] * n

    for start in range(n):
        if not adj_list[start]:
            continue
        # BFS to find a cycle
        visited[start] = True
        parent[start] = -1
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle: reconstruct
                    cycle = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.append(v)
                    # Reverse to get proper order
                    cycle = cycle[::-1]
                    # Remove duplicate endpoint
                    if cycle[0] == cycle[-1]:
                        cycle = cycle[:-1]
                    if len(cycle) >= 3:
                        return cycle
        visited = [False] * n
        parent = [-1] * n

    return None


def _decompose(edges, n, deadline):
    """Decompose edge set into cycles and single edges."""
    if not edges:
        return []

    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    rng = random.Random(99999)

    while remaining:
        if time.time() > deadline:
            break

        # Build adjacency list from remaining edges
        adj_list = defaultdict(list)
        for u, v in remaining:
            adj_list[u].append(v)
            adj_list[v].append(u)

        # Find a cycle
        cycle = _find_cycle_in_subgraph(adj_list, n, rng)
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
        else:
            # No cycle; take one edge
            e = next(iter(remaining))
            remaining.discard(e)
            pieces.append([e[0], e[1]])

    if remaining:
        for e in remaining:
            pieces.append([e[0], e[1]])

    return pieces


def construct():
    """Return three lists of pieces, one per fixed instance graph."""
    t_start = time.time()
    deadline = t_start + 140.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [_instance(*spec) for spec in specs]

    results = []
    for i, (edges, spec) in enumerate(zip(graphs, specs)):
        n = spec[0]
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
        if time.time() > deadline:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

# EVOLVE-BLOCK-END
