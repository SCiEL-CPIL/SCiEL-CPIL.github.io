"""Decompose three fixed ER graphs into cycles and edges to minimize piece count."""
import time
import random
import sys
from collections import defaultdict

# Deadline guard: evaluator kills at 150s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 140.0  # 10s margin

def _time_left():
    return _DEADLINE - time.time()

def instance_graph(n, p, seed):
    """Reproduce the verifier's instance generator exactly."""
    rng = Random = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    # Path edges from shuffled order
    for i in range(n - 1):
        u, v = order[i], order[i + 1]
        edges.add((min(u, v), max(u, v)))
    # Random edges
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges and rng.random() < p:
                edges.add((u, v))
    return sorted(edges)

def _build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_long_cycle(adj, edges_set, start):
    """
    Find a long cycle starting from `start` using a greedy/backtracking heuristic.
    Returns a list of vertices forming the cycle, or None.
    """
    n = len(adj)
    best_cycle = None
    
    def dfs(path, used_edges, visited):
        nonlocal best_cycle
        if len(path) > 3:
            # Check if we can close the cycle
            if path[0] in adj[path[-1]] and (min(path[0], path[-1]), max(path[0], path[-1])) not in used_edges:
                cycle = path[:]
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle[:]
        
        if _time_left() < 0.5:
            return
        
        current = path[-1]
        # Try neighbors in order of degree (prefer lower degree for longer paths, or higher?)
        # Actually for finding long cycles, we want to avoid dead-ends.
        # Let's try a simple heuristic: prefer neighbors with higher remaining degree.
        neighbors = sorted(adj[current], key=lambda x: len([e for e in adj[x] if (min(x,e), max(x,e)) not in used_edges]), reverse=True)
        
        for nb in neighbors:
            if nb == path[-2] if len(path) >= 2 else False:
                continue  # Don't go back
            if (min(current, nb), max(current, nb)) in used_edges:
                continue
            if nb in visited and nb != path[0]:
                continue
            
            new_path = path + [nb]
            new_used = used_edges | {(min(current, nb), max(current, nb))}
            new_visited = visited | {nb}
            
            dfs(new_path, new_used, new_visited)
            
            if best_cycle is not None and len(best_cycle) >= n * 0.5:
                return  # Good enough, stop searching
    
    dfs([start], set(), {start})
    return best_cycle

def _decompose_graph(edges):
    """
    Decompose a graph into cycles and edges.
    Strategy: Repeatedly find a long cycle, remove its edges, repeat.
    Remaining edges become single-edge pieces.
    """
    if not edges:
        return []
    
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    adj = _build_adjacency(edges_set)
    pieces = []
    
    # Work with a mutable copy of edges
    remaining = set(edges_set)
    
    # While there are edges and time remains
    while remaining and _time_left() > 1.0:
        # Find a vertex with max degree to start from
        if not remaining:
            break
        
        # Rebuild adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find a vertex with degree >= 2 (part of a cycle)
        cycle_vertices = [v for v in rem_adj if len(rem_adj[v]) >= 2]
        if not cycle_vertices:
            break  # No more cycles possible
        
        # Pick vertex with max degree
        start = max(cycle_vertices, key=lambda v: len(rem_adj[v]))
        
        # Find a long cycle
        cycle = _find_long_cycle(rem_adj, remaining, start)
        
        if cycle and len(cycle) >= 3:
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                remaining.discard(edge)
            pieces.append(cycle)
        else:
            # No cycle found; break to avoid infinite loop
            break
    
    # Remaining edges become single-edge pieces
    for u, v in sorted(remaining):
        pieces.append([u, v])
    
    return pieces

def _validate_decomposition(edges, pieces):
    """Validate that pieces are edge-disjoint and cover all edges."""
    edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    used_edges = set()
    
    for piece in pieces:
        if len(piece) == 2:
            # Single edge
            u, v = piece
            edge = (min(u, v), max(u, v))
            if edge in used_edges:
                return False
            if edge not in edges_set:
                return False
            used_edges.add(edge)
        elif len(piece) >= 3:
            # Cycle
            # Check distinct vertices
            if len(set(piece)) != len(piece):
                return False
            for i in range(len(piece)):
                u, v = piece[i], piece[(i + 1) % len(piece)]
                edge = (min(u, v), max(u, v))
                if edge in used_edges:
                    return False
                if edge not in edges_set:
                    return False
                used_edges.add(edge)
        else:
            return False
    
    return used_edges == edges_set

def construct():
    """
    Construct decompositions for three fixed graphs.
    Returns three lists of pieces.
    """
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for n, p, seed in specs:
        if _time_left() < 1.0:
            # Not enough time; fall back to trivial decomposition
            edges = instance_graph(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance_graph(n, p, seed)
        
        # Try decomposition
        pieces = _decompose_graph(edges)
        
        # Validate; if invalid, fall back to trivial
        if not _validate_decomposition(edges, pieces):
            pieces = [[u, v] for u, v in edges]
        
        results.append(pieces)
    
    return results
