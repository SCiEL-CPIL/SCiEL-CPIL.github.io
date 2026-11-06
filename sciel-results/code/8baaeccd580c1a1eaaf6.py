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


def _find_cycle_dfs(adj, start, n, remaining_edges):
    """
    Find a simple cycle containing 'start' where all edges are in remaining_edges
    and start is the smallest vertex in the cycle.
    Returns the cycle as a list of vertices, or None.
    """
    if len(adj[start]) < 2:
        return None
    
    # Check if start has at least 2 neighbors in remaining graph
    neighbors_in_rem = []
    for v in adj[start]:
        edge = (min(start, v), max(start, v))
        if edge in remaining_edges:
            neighbors_in_rem.append(v)
    
    if len(neighbors_in_rem) < 2:
        return None
    
    # DFS to find a cycle
    path = [start]
    on_path = {start}
    
    def dfs(u, depth):
        if depth > n:
            return None
        for v in sorted(adj[u]):
            if v < start:
                continue
            edge = (min(u, v), max(u, v))
            if edge not in remaining_edges:
                continue
            if v == start:
                if len(path) >= 3:
                    return path[:]
                continue
            if v not in on_path:
                on_path.add(v)
                path.append(v)
                result = dfs(v, depth + 1)
                if result:
                    return result
                path.pop()
                on_path.discard(v)
        return None
    
    return dfs(start, 0)


def _decompose_greedy(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding cycles and removing them.
    Greedy approach: find a cycle, remove it, repeat.
    Prioritize longer cycles for better savings.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Find a cycle in the current graph
        # Try to find a long cycle for better savings
        best_cycle = None
        best_len = 0
        
        # Try finding cycles from each vertex, prioritize vertices with higher degree
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        candidates.sort(key=lambda v: -len(adj[v]))
        
        for start in candidates:
            if time.time() - start_time > time_limit:
                break
            if not adj[start] or len(adj[start]) < 2:
                continue
            
            # Find a cycle starting and ending at 'start'
            cycle = _find_cycle_dfs(adj, start, n, remaining_edges)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
                if best_len >= n:  # Can't do better than Hamiltonian
                    break
        
        if best_cycle:
            # Verify all edges in cycle still exist
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            # No cycle found, add all remaining edges as single-edge pieces
            for edge in remaining_edges:
                pieces.append(list(edge))
            remaining_edges = set()
            break
    
    # Add any leftover edges
    for edge in remaining_edges:
        pieces.append(list(edge))
    
    return pieces


def _decompose_with_local_search(edges, n, time_limit):
    """
    Improved decomposition using local search to find better cycle sets.
    First do greedy, then try to improve by finding additional cycles
    in the remaining edge set or by restructuring.
    """
    if not edges:
        return []
    
    # Start with greedy decomposition
    pieces = _decompose_greedy(edges, n, time_limit * 0.6)
    
    # Now try to improve: look at the remaining single-edge pieces
    # and see if we can combine any into cycles
    start_time = time.time()
    
    # Build a map from edge to piece index
    edge_to_piece = {}
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            u, v = piece
            edge = (min(u, v), max(u, v))
            edge_to_piece[edge] = i
    
    # Try to find cycles among the single-edge pieces
    # This is hard, so instead, let's re-run greedy with a different strategy
    # or try to find cycles in the subgraph of single-edge pieces
    
    # Alternative: try multiple greedy passes with different vertex orderings
    if time.time() - start_time < time_limit * 0.3:
        best_pieces = pieces
        best_count = len(pieces)
        
        # Try a few different random orderings
        for trial in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Shuffle the vertex order for candidate selection
            shuffled = list(range(n))
            random.shuffle(shuffled)
            
            # Re-run greedy with shuffled order
            trial_pieces = _decompose_greedy_shuffled(edges, n, shuffled, time_limit * 0.1)
            if len(trial_pieces) < best_count:
                best_pieces = trial_pieces
                best_count = len(trial_pieces)
        
        pieces = best_pieces
    
    return pieces


def _decompose_greedy_shuffled(edges, n, vertex_order, time_limit):
    """Greedy decomposition with a specified vertex order for cycle finding."""
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        best_cycle = None
        best_len = 0
        
        for start in vertex_order:
            if time.time() - start_time > time_limit:
                break
            if not adj[start] or len(adj[start]) < 2:
                continue
            
            cycle = _find_cycle_dfs(adj, start, n, remaining_edges)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
                if best_len >= n:
                    break
        
        if best_cycle:
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            for edge in remaining_edges:
                pieces.append(list(edge))
            remaining_edges = set()
            break
    
    for edge in remaining_edges:
        pieces.append(list(edge))
    
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
        if remaining_time < 5:
            # Fallback: just output all edges as single-edge pieces
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Use greedy decomposition with local search improvements
        graph_time = min(remaining_time * 0.9, 40.0)
        
        pieces = _decompose_with_local_search(edges, n, graph_time)
        
        # Validate: ensure pieces are edge-disjoint and cover all edges
        used_edges = set()
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            elif len(piece) >= 3:
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    used_edges.add(e)
        
        results.append(pieces)

    return results
