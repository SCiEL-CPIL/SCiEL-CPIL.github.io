"""Decompose three fixed graphs into cycles + edges, minimizing piece count."""
import time
import random
import sys
from collections import defaultdict, deque

_DEADLINE = 120.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

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

def _find_longest_path_cycle(adj, start, max_depth=60):
    """DFS to find a long simple path starting at start; if it ends at a neighbor of start, close to cycle."""
    best = None
    best_len = 0
    sys.setrecursionlimit(10000)
    def dfs(node, visited, path):
        nonlocal best, best_len
        if len(path) > best_len:
            # check if can close
            if len(path) >= 3 and path[-1] in adj.get(start, set()):
                best = list(path)
                best_len = len(path)
        if len(path) >= max_depth:
            return
        for nb in sorted(adj.get(node, set())):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(nb, visited, path)
                path.pop()
                visited.remove(nb)
    visited = {start}
    path = [start]
    dfs(start, visited, path)
    return best

def _greedy_decompose(edges, strategy=0, rng=None):
    """
    Greedy cycle extraction. Returns list of pieces.
    strategy 0: pick longest cycle found by DFS from each vertex
    strategy 1: pick cycle through highest-degree vertex first
    strategy 2: use 2-core peeling order
    """
    if rng is None:
        rng = random.Random(0)
    remaining = set(edges)
    if not remaining:
        return []
    adj = _build_adj(remaining)
    pieces = []
    
    while remaining:
        if _remaining() < 1.0:
            break
        # Find a cycle in the current graph
        cycle = None
        if strategy == 0:
            # Try each vertex, find longest cycle through it
            best_cycle = None
            best_len = 0
            for v in list(adj.keys()):
                c = _find_longest_path_cycle(adj, v, max_depth=50)
                if c and len(c) > best_len:
                    best_len = len(c)
                    best_cycle = c
            if best_cycle:
                cycle = best_cycle
        elif strategy == 1:
            # Start from highest degree vertex
            if adj:
                start = max(adj.keys(), key=lambda v: len(adj[v]))
                cycle = _find_longest_path_cycle(adj, start, max_depth=50)
                if cycle is None:
                    # try second highest
                    degs = sorted(adj.keys(), key=lambda v: -len(adj[v]))
                    for s in degs[:5]:
                        c = _find_longest_path_cycle(adj, s, max_depth=50)
                        if c:
                            cycle = c
                            break
        elif strategy == 2:
            # 2-core: repeatedly remove degree-1 vertices, then find cycle in core
            core_adj = {v: set(nbs) for v, nbs in adj.items()}
            queue = deque([v for v in core_adj if len(core_adj[v]) < 2])
            while queue:
                v = queue.popleft()
                if v not in core_adj:
                    continue
                if len(core_adj[v]) >= 2:
                    continue
                for nb in list(core_adj[v]):
                    core_adj[nb].discard(v)
                    if len(core_adj[nb]) < 2 and nb in core_adj:
                        queue.append(nb)
                del core_adj[v]
            if core_adj:
                start = next(iter(core_adj))
                cycle = _find_longest_path_cycle(core_adj, start, max_depth=50)
                if cycle is None:
                    for s in core_adj:
                        c = _find_longest_path_cycle(core_adj, s, max_depth=50)
                        if c:
                            cycle = c
                            break
        
        if cycle is None or len(cycle) < 3:
            # No cycle found; take an edge
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.discard(e)
            # update adj
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
            continue
        
        # Extract cycle edges
        cycle_edges = set()
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            ce = (min(a, b), max(a, b))
            if ce in remaining:
                cycle_edges.add(ce)
            else:
                # edge not in remaining, skip this cycle attempt
                cycle = None
                break
        
        if cycle is None or len(cycle_edges) < 3:
            # Fallback: take an edge
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
            continue
        
        pieces.append(list(cycle))
        for ce in cycle_edges:
            remaining.discard(ce)
            u, v = ce
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
    
    # Remaining edges as individual pieces
    for e in sorted(remaining):
        pieces.append([e[0], e[1]])
    
    return pieces

def _validate(pieces, edges):
    """Check that pieces are edge-disjoint and cover all edges."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    used = set()
    for p in pieces:
        if len(p) == 2:
            e = tuple(sorted(p))
            if e in used:
                return False
            used.add(e)
        elif len(p) >= 3:
            # Check all vertices distinct
            if len(set(p)) != len(p):
                return False
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                e = tuple(sorted((a, b)))
                if e in used:
                    return False
                used.add(e)
        else:
            return False
    return used == edge_set

def _local_improve(pieces, edges, time_budget=2.0):
    """Try to merge two edge-pieces into a cycle if possible, or extend cycles."""
    # This is a simple improvement: look for two edges that form a path of length 2
    # where the middle vertex connects to both endpoints, forming a triangle
    # For now, just return as-is; improvement is complex
    return pieces

def construct():
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    rng = random.Random(42)
    
    for idx, spec in enumerate(specs):
        if _remaining() < 5.0:
            # Not enough time; just do edges
            edges = _instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(*spec)
        best_pieces = None
        best_count = len(edges)  # baseline: all edges
        
        # Try multiple strategies and random seeds
        for strategy in [0, 1, 2]:
            for attempt in range(3):
                if _remaining() < 3.0:
                    break
                r = random.Random(idx * 100 + strategy * 10 + attempt)
                pieces = _greedy_decompose(edges, strategy=strategy, rng=r)
                if not _validate(pieces, edges):
                    continue
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
            if _remaining() < 3.0:
                break
        
        if best_pieces is None:
            best_pieces = [[u, v] for u, v in edges]
        
        results.append(best_pieces)
    
    return results
