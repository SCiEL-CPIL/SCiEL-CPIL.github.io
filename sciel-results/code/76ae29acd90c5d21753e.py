"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0  # Safety margin from 150s limit


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycle_iterative(adj, start, n, remaining_edges):
    """
    Find one simple cycle containing 'start' where start is the smallest vertex,
    using only edges in remaining_edges. Iterative DFS with explicit stack.
    Returns list of vertices or None.
    """
    if not adj[start]:
        return None
    
    # Check that start has at least 2 neighbors in remaining edges
    start_neighbors = [v for v in adj[start] 
                       if v >= start and (min(start, v), max(start, v)) in remaining_edges]
    if len(start_neighbors) < 2:
        return None
    
    # Iterative DFS
    # Stack entries: (vertex, next_neighbor_index, path_so_far)
    path = [start]
    visited = {start}
    # For each vertex in path, track which neighbor index we're at
    neighbor_indices = [0]
    
    # Precompute sorted neighbor lists filtered by remaining edges and v >= start
    def get_valid_neighbors(u):
        result = []
        for v in sorted(adj[u]):
            if v < start:
                continue
            edge = (min(u, v), max(u, v))
            if edge not in remaining_edges:
                continue
            result.append(v)
        return result
    
    valid_neighbors = {}
    for v in [start]:
        valid_neighbors[v] = get_valid_neighbors(v)
    
    while True:
        u = path[-1]
        if u not in valid_neighbors:
            valid_neighbors[u] = get_valid_neighbors(u)
        
        nbrs = valid_neighbors[u]
        idx = neighbor_indices[-1]
        
        found = False
        while idx < len(nbrs):
            v = nbrs[idx]
            neighbor_indices[-1] = idx + 1
            if v == start:
                if len(path) >= 3:
                    return path[:]
            elif v not in visited:
                visited.add(v)
                path.append(v)
                neighbor_indices.append(0)
                found = True
                break
            idx += 1
        
        if found:
            continue
        
        # Backtrack
        if len(path) == 1:
            return None
        visited.discard(path[-1])
        path.pop()
        neighbor_indices.pop()


def _decompose_max_cycles(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding cycles and removing them.
    Strategy: at each step, find a cycle that maximizes the number of edges saved.
    A cycle of length k saves k-1 pieces compared to using k single edges.
    We use a greedy approach: find the longest cycle available.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
        
        # Find the best (longest) cycle in the remaining graph
        best_cycle = None
        best_len = 0
        
        # Try each vertex as potential start (smallest in cycle)
        for start in range(n):
            if time.time() - start_time > time_limit:
                break
            if len(adj[start]) < 2:
                continue
            
            # Quick check: does start have at least 2 neighbors in remaining?
            count = 0
            for v in adj[start]:
                if v >= start and (min(start, v), max(start, v)) in remaining:
                    count += 1
                    if count >= 2:
                        break
            if count < 2:
                continue
            
            # Try to find a cycle starting from 'start'
            cycle = _find_cycle_iterative(adj, start, n, remaining)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
                if best_len >= n:
                    break
        
        if best_cycle:
            # Validate all edges still exist
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            # No more cycles, add all remaining as single edges
            for edge in remaining:
                pieces.append(list(edge))
            remaining = set()
            break
    
    # Add any leftover
    for edge in remaining:
        pieces.append(list(edge))
    
    return pieces


def _decompose_with_improvement(edges, n, time_limit):
    """
    Decompose by finding cycles greedily, then attempt local improvement.
    First pass: greedy longest-cycle extraction.
    Second pass: try to merge adjacent single-edge pieces into cycles.
    """
    pieces = _decompose_max_cycles(edges, n, time_limit * 0.6)
    
    # Local improvement: try to find cycles among the single-edge pieces
    # combined with some cycle edges that we might re-extract
    # This is complex, so instead do a second greedy pass on remaining structure
    
    # Actually, let's try a different strategy: 
    # Count how many single edges we have, and try to find cycles that use some of them
    # For simplicity, just return the greedy result
    return pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 10:
            # Fallback: just output all edges as single-edge pieces
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Time budget for this graph
        graph_time = min(remaining_time * 0.9, 45.0)
        
        pieces = _decompose_max_cycles(edges, n, graph_time)
        
        # Validate: ensure pieces are edge-disjoint and cover all edges
        used_edges = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check distinct vertices
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
        
        if not valid:
            # Fallback to trivial decomposition
            results.append([[u, v] for u, v in edges])
        else:
            results.append(pieces)

    return results
