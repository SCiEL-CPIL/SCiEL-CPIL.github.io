import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, rnd, max_attempts=30):
    """
    Find a long simple cycle using randomized greedy path extension with backtracking.
    Returns list of vertices (no closing duplicate) or None.
    """
    best_cycle = None
    best_len = 0

    for attempt in range(max_attempts):
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rnd.choice(candidates)

        # DFS with backtracking to find a long cycle
        path = [start]
        visited = {start}
        current = start
        stack = [(start, list(adj[start]))]

        while stack:
            current, neighbors = stack[-1]

            # Check if we can close the cycle
            if start in neighbors and len(path) >= 3:
                cycle = path[:]
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                # Try to find even longer cycles
                neighbors.remove(start)
                if not neighbors:
                    stack.pop()
                    path.pop()
                    visited.discard(current)
                continue

            # Find next unvisited neighbor
            next_v = None
            while neighbors:
                v = neighbors.pop(0)
                if v not in visited:
                    next_v = v
                    break

            if next_v is None:
                # Dead end, backtrack
                stack.pop()
                path.pop()
                visited.discard(current)
            else:
                # Extend path
                path.append(next_v)
                visited.add(next_v)
                stack.append((next_v, list(adj[next_v])))

        # If we found a cycle, we could stop early, but let's try more
        # Actually, for efficiency, let's just return the best found so far
        if best_cycle and best_len >= n * 0.7:
            break

    return best_cycle

def _decompose_greedy(edges, n, rnd, time_limit=None):
    """
    Greedy decomposition: repeatedly find a long cycle, remove it, until no cycles remain.
    Remaining edges become single-edge pieces.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    start_time = time.time()

    while remaining > 0:
        if time_limit and time.time() > time_limit:
            break

        # Find a cycle using randomized DFS
        cycle = _find_long_cycle(adj, n, rnd)

        if cycle:
            # Remove cycle edges
            m = len(cycle)
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cycle)
        else:
            # No cycle found. Remove edges as single-edge pieces.
            # Find all remaining edges
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        pieces.append([u, v])
                        remaining -= 1
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    # Leftover edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    return pieces

def _local_search_polish(pieces, edges_set, n, rnd, iterations=50):
    """
    Try to improve the decomposition by merging single-edge pieces into cycles.
    This is a simplified local search.
    """
    # This is a placeholder for a more sophisticated local search
    # For now, just return the pieces as-is
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    base_seed = 42
    attempt = 0

    # We will run multiple restarts with different random seeds
    # and keep the best result
    while time.time() < deadline - 0.5:
        rnd = random.Random(base_seed + attempt)
        time_limit = deadline - 0.5
        pieces = _decompose_greedy(edges, n, rnd, time_limit)
        cnt = len(pieces)
        if cnt < best_count:
            best = pieces
            best_count = cnt

        attempt += 1
        # Stop if we've tried enough or time is running out
        if attempt > 100:
            break

    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Give more time to larger graphs if needed, but keep balanced
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    return results
