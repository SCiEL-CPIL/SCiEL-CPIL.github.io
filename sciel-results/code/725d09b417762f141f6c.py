"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
import time
import random
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _start_time)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle(adj, edge_set, rng, max_nodes=200):
    """Try to find a long simple cycle using Warnsdorff-ordered DFS with multiple starts."""
    best_cycle = None
    n = len(adj)
    if n < 3:
        return None

    # Try multiple random start vertices
    starts = list(adj.keys())
    rng.shuffle(starts)
    for start in starts[:min(len(starts), 10)]:
        if len(edge_set) < 3:
            break
        # DFS to find a simple path from start, trying to close back to start
        # State: path (list of vertices), visited set
        # We'll do iterative deepening with ordering
        best_local = None

        def dfs(path, visited):
            nonlocal best_local
            if len(path) > max_nodes:
                return
            cur = path[-1]
            # Try to close cycle
            if len(path) >= 3 and start in adj[cur] and (min(cur, start), max(cur, start)) in edge_set:
                cycle = list(path)
                if best_local is None or len(cycle) > len(best_local):
                    best_local = cycle
                # Don't return; maybe we can find even longer
            # Order neighbors: prefer those with smallest remaining degree (Warnsdorff)
            neighbors = []
            for nb in adj[cur]:
                if nb in visited and nb != start:
                    continue
                if nb == start and len(path) < 3:
                    continue
                # Edge must be in edge_set
                if (min(cur, nb), max(cur, nb)) not in edge_set:
                    continue
                # Count remaining unvisited neighbors for nb (excluding path-internal)
                rem_deg = sum(1 for x in adj[nb] if x not in visited or x == start)
                neighbors.append((rem_deg, nb))
            neighbors.sort(key=lambda x: x[0])  # smallest remaining degree first
            for _, nb in neighbors:
                if nb == start:
                    # Only close if length >= 3
                    if len(path) >= 3:
                        cycle = list(path)
                        if best_local is None or len(cycle) > len(best_local):
                            best_local = cycle
                        continue
                visited.add(nb)
                path.append(nb)
                dfs(path, visited)
                path.pop()
                visited.remove(nb)

        dfs([start], {start})
        if best_local is not None:
            if best_cycle is None or len(best_local) > len(best_cycle):
                best_cycle = best_local

    return best_cycle


def _decompose_one(edges):
    """Decompose a set of edges into cycles and single edges, minimizing piece count."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    if not edge_set:
        return []

    cycles = []
    leftover = set(edge_set)
    rng = random.Random(42)

    while leftover and _remaining() > 5.0:
        adj = _build_adj(leftover)
        # Find a cycle
        cycle = _find_long_cycle(adj, leftover, rng)
        if cycle is None:
            # Try a simpler 3-cycle search
            cycle = _find_3cycle(leftover)
        if cycle is None:
            break

        # Remove cycle edges
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in leftover:
                leftover.remove(e)
        cycles.append(cycle)

    # Leftover edges as single-edge pieces
    pieces = [list(c) for c in cycles]
    for e in leftover:
        pieces.append(list(e))
    return pieces


def _find_3cycle(edge_set):
    """Find any triangle in the edge set."""
    adj = _build_adj(edge_set)
    for u in adj:
        for v in adj[u]:
            if u < v and (u, v) in edge_set:
                for w in adj[u]:
                    if w > v and (u, w) in edge_set and (v, w) in edge_set:
                        return [u, v, w]
    return None


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if _remaining() < 10.0:
            # Fallback: return all edges as single pieces for remaining
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        pieces = _decompose_one(edges)
        results.append(pieces)
    return results
