"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_shortest_cycle(adj, start, n):
    """Find shortest cycle containing `start` using BFS from neighbors."""
    best_cycle = None
    best_len = n + 1
    # BFS from each neighbor of start
    for nb in adj[start]:
        # BFS from nb, looking for path back to start
        visited = {start: None, nb: nb}
        queue = deque([nb])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == start:
                    # Found a cycle: start -> ... -> u -> start
                    # Reconstruct path from nb to u
                    path = [u]
                    cur = u
                    while cur != nb:
                        cur = visited[cur]
                        path.append(cur)
                    path.reverse()
                    cycle = [start] + path
                    if len(cycle) < best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    break
                if v not in visited:
                    visited[v] = u
                    queue.append(v)
    return best_cycle

def _find_cycle_greedy(adj, n):
    """Find a cycle by starting from high-degree vertices."""
    best_cycle = None
    best_len = n + 1
    
    # Try vertices in order of decreasing degree
    vertices = sorted(range(n), key=lambda v: -len(adj[v]))
    
    for start in vertices:
        if not adj[start]:
            continue
        cycle = _find_shortest_cycle(adj, start, n)
        if cycle is not None and len(cycle) < best_len:
            best_len = len(cycle)
            best_cycle = cycle
            if best_len <= 3:
                break
    
    return best_cycle

def _decompose(edges, n):
    """Decompose edge set into cycles and single edges."""
    pieces = []
    # Build adjacency
    adj = defaultdict(set)
    remaining = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        remaining.add((u, v) if u < v else (v, u))
    
    # Greedily extract cycles
    max_iterations = 1000
    it = 0
    while remaining and it < max_iterations:
        it += 1
        cycle = _find_cycle_greedy(adj, n)
        if cycle is None:
            break
        # Remove cycle edges from remaining
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (u, v) if u < v else (v, u)
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
        pieces.append(cycle)
    
    # Remaining edges as single pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def _local_search_improve(pieces, edges_set, n, max_rounds=50):
    """Try to improve decomposition by merging or re-extracting."""
    # For now, just return pieces as-is
    # Future: try to find cycles in the "residual" graph formed by
    # endpoints of single-edge pieces
    return pieces

def construct():
    deadline = time.time() + 120  # Safety margin before 150s limit
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
        
        if time.time() > deadline - 5:
            # Not enough time for remaining instances; use trivial decomposition
            for n2, p2, seed2 in specs[len(results):]:
                edges2 = _instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
