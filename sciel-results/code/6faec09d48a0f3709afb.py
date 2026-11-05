"""Cycle decomposition via greedy long-cycle extraction with time budget."""
import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    """Decompose edge set into cycles and single edges. Minimize piece count."""
    n = max((max(e) for e in edges), default=0) + 1
    # Build adjacency as dict of sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    pieces = []
    # Greedily extract cycles: find a cycle, remove its edges, repeat
    # Prefer longer cycles to minimize piece count
    while edge_set:
        if time.time() > deadline:
            break
        # Find a cycle via DFS from a vertex with highest remaining degree
        # Try to find a reasonably long cycle
        start_candidates = sorted(range(n), key=lambda x: -len(adj[x]))
        found_cycle = None
        for start in start_candidates:
            if not adj[start]:
                continue
            # DFS to find a cycle back to start
            # Use a simple DFS with path tracking
            visited = {start}
            path = [start]
            parent = {start: None}
            # Iterative DFS
            stack = [start]
            found = False
            while stack and not found:
                node = stack[-1]
                neighbors = list(adj[node])
                # Try neighbors
                advanced = False
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        # Found cycle
                        found_cycle = list(path)
                        found = True
                        break
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        parent[nb] = node
                        stack.append(nb)
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    if path:
                        path.pop()
                        if len(visited) > 0 and path:
                            pass
                    if path:
                        visited.discard(path[-1] if path else start)
            if found:
                break
        if found_cycle is None:
            # No cycle found from any start; take remaining edges as singles
            for (u, v) in list(edge_set):
                pieces.append([u, v])
            edge_set.clear()
            break
        # Remove cycle edges
        cycle = found_cycle
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            e = (min(a, b), max(a, b))
            if e in edge_set:
                edge_set.remove(e)
                adj[a].discard(b)
                adj[b].discard(a)
        pieces.append(cycle)

    # If we hit deadline, add remaining edges as singles
    for (u, v) in list(edge_set):
        pieces.append([u, v])
    return pieces


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        if time.time() > deadline:
            # Fallback: all edges as singles
            results.append([list(e) for e in edges])
            continue
        pieces = _decompose(edges, deadline)
        results.append(pieces)
    return results
