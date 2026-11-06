"""
Decomposition of three fixed graphs into few cycles and edges.

Strategy: randomized contraction-based heuristic combined with greedy cycle extraction.
1. Build the graph.
2. Repeatedly identify high-degree 2-connected components (or dense subgraphs) and
   extract long cycles from them, removing those edges.
3. Use a degree-sorted DFS to find longer cycles in the remaining graph.
4. Remaining edges become single-edge pieces.
"""

import time
import random
from collections import defaultdict, deque
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle(adj, edges_set, start_vertex=None, max_len=None):
    """
    Find a long cycle starting from a given vertex using DFS with backtracking.
    Returns a list of vertices forming the cycle, or None if no cycle found.
    """
    if start_vertex is None:
        return None
    
    n = max(adj.keys()) + 1 if adj else 0
    visited = set()
    path = []
    
    def dfs(v, depth):
        if max_len is not None and depth >= max_len:
            return None
        if v in visited:
            return None
        visited.add(v)
        path.append(v)
        
        for neighbor in sorted(adj[v]):
            if neighbor == start_vertex and len(path) >= 3:
                return path[:]
            if neighbor not in visited:
                result = dfs(neighbor, depth + 1)
                if result is not None:
                    return result
        
        path.pop()
        visited.remove(v)
        return None
    
    return dfs(start_vertex, 0)


def _find_cycle_greedy(adj, edges_set):
    """
    Greedily find a cycle by following edges, preferring high-degree vertices.
    """
    if not edges_set:
        return None
    
    # Start from a vertex with high degree
    start = max(adj.keys(), key=lambda v: len(adj[v])) if adj else None
    if start is None:
        return None
    
    path = [start]
    visited = {start}
    current = start
    
    while True:
        found_next = False
        # Try neighbors in order of decreasing degree
        neighbors = sorted([n for n in adj[current] if (min(current, n), max(current, n)) in edges_set],
                          key=lambda x: -len(adj[x]))
        
        for nxt in neighbors:
            if nxt == start and len(path) >= 3:
                return path
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                current = nxt
                found_next = True
                break
        
        if not found_next:
            return None


def _find_cycles_in_graph(adj, edges_set, time_limit):
    """
    Find multiple cycles in the graph using greedy extraction.
    Returns a list of cycles (each a list of vertices).
    """
    cycles = []
    remaining_edges = set(edges_set)
    
    while remaining_edges and time.time() < time_limit:
        # Find a vertex with the highest degree in remaining graph
        deg_in_remaining = defaultdict(int)
        for u, v in remaining_edges:
            deg_in_remaining[u] += 1
            deg_in_remaining[v] += 1
        
        if not deg_in_remaining:
            break
        
        start = max(deg_in_remaining.keys(), key=lambda v: deg_in_remaining[v])
        
        # Build adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Try to find a cycle starting from start
        cycle = _find_cycle_greedy(rem_adj, remaining_edges)
        
        if cycle is None:
            # Try DFS-based approach
            cycle = _find_long_cycle(rem_adj, remaining_edges, start)
        
        if cycle and len(cycle) >= 3:
            # Remove edges of this cycle from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
            cycles.append(cycle)
        else:
            # No more cycles can be found
            break
    
    return cycles, remaining_edges


def _decompose_graph(edges):
    """
    Decompose a graph into cycles and single edges.
    """
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adjacency(edges)
    
    # Phase 1: Contraction-based cycle extraction
    # Identify dense subgraphs and extract cycles from them
    deadline = time.time() + 5  # 5 seconds for this graph
    
    cycles, remaining_edges = _find_cycles_in_graph(adj, edges_set, deadline)
    
    # Phase 2: Try to find more cycles in remaining edges
    if remaining_edges and time.time() < deadline:
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        extra_cycles, remaining_edges = _find_cycles_in_graph(rem_adj, remaining_edges, deadline)
        cycles.extend(extra_cycles)
    
    # Build pieces
    pieces = []
    
    # Add cycles as pieces
    for cycle in cycles:
        # Convert to list of vertices (the closing edge is implied)
        pieces.append(cycle)
    
    # Add remaining edges as single-edge pieces
    for u, v in remaining_edges:
        pieces.append([u, v])
    
    return pieces


def construct():
    """
    Decompose three fixed instance graphs into cycles and single edges.
    Returns three lists of pieces.
    """
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    
    return results
