import time
import random
from collections import defaultdict, deque

_T0 = time.time()
_DEADLINE = 145.0  # Wall-clock limit with margin (evaluator kills at 150s)

def _instance(n, p, seed):
    """
    Reproduce the instance generation logic from the task description.
    """
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(adj, n, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    for p in pieces:
        k = len(p)
        if k == 2:
            e = tuple(sorted(p))
            if e in seen:
                return False
            seen.add(e)
        elif k >= 3:
            for i in range(k):
                e = tuple(sorted((p[i], p[(i + 1) % k])))
                if e in seen:
                    return False
                seen.add(e)
        else:
            return False
    all_edges = set()
    for v in range(n):
        for w in adj[v]:
            if v < w:
                all_edges.add((v, w))
    return seen == all_edges

def _find_longest_cycle_in_path(path, adj, deadline):
    """
    Given a simple path, find the longest cycle that can be formed by adding one edge.
    Returns the cycle as a list of vertices, or None.
    """
    best_cycle = None
    best_len = 0
    L = len(path)
    
    # Check closing edge
    if L >= 3 and path[-1] in adj[path[0]]:
        best_cycle = list(path)
        best_len = L
    
    # Check back edges
    # We want the longest cycle, so we look for the longest suffix of the path
    # that forms a cycle with an earlier vertex.
    # Actually, any pair (i, j) with i < j and edge (path[i], path[j]) forms a cycle path[i:j+1].
    # To find the longest, we should iterate i from 0 to L-3, and j from L-1 down to i+2.
    # The first valid j for a given i gives the longest cycle starting at i.
    # We compare lengths across all i.
    
    for i in range(L - 2):
        if time.time() > deadline:
            break
        # If the current best cycle is already longer than any possible cycle starting at i, skip?
        # Max length starting at i is L - i. If best_len >= L - i, we can break?
        # No, because a cycle starting at i+1 could be longer if the closing edge is further back?
        # No, the cycle is a contiguous subpath.
        # The longest possible cycle starting at i is path[i:L] if path[i] connects to path[L-1].
        # So if best_len >= L - i, no cycle starting at i can be longer.
        if best_len >= L - i:
            break
            
        for j in range(L - 1, i + 1, -1):
            if path[i] in adj[path[j]]:
                cycle = path[i:j+1]
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                break # Found the longest cycle for this i
            
    return best_cycle

def _find_cycle_greedy(adj, n, start, deadline, rng, strategy=0):
    """
    Find a long cycle starting from 'start'.
    """
    path = [start]
    visited = {start}
    current = start
    
    while True:
        if time.time() > deadline:
            break
        
        unvisited_neighbors = [w for w in adj[current] if w not in visited]
        if not unvisited_neighbors:
            break
        
        if strategy == 0:
            # Standard greedy: highest degree
            unvisited_neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
            k = min(3, len(unvisited_neighbors))
            next_node = rng.choice(unvisited_neighbors[:k])
        elif strategy == 1:
            # Prefer neighbors that can close the cycle if we have a decent path length
            scored = []
            for w in unvisited_neighbors:
                score = len(adj[w])
                if w in adj[start] and len(path) >= 4:
                    score += 100  # Strong incentive to close if path is long enough
                scored.append((score, w))
            scored.sort(reverse=True)
            k = min(3, len(scored))
            _, next_node = rng.choice(scored[:k])
        else:
            # Random
            next_node = rng.choice(unvisited_neighbors)
        
        visited.add(next_node)
        path.append(next_node)
        current = next_node
        
        # Early stop if we hit a very long path?
        if len(path) >= n:
            break
            
    # Find the longest cycle in the generated path
    return _find_longest_cycle_in_path(path, adj, deadline)

def _decompose_graph(n, edges, deadline, seed_offset=0):
    """
    Decompose the graph into cycles and single edges.
    Returns a list of pieces.
    """
    adj = _adj_from_edges(edges, n)
    remaining_edges = set(edges)
    pieces = []
    work_adj = [set(s) for s in adj]
    
    # We will try multiple strategies for cycle finding
    strategies = [0, 1, 2]
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if any edges remain
        has_edge = False
        for v in range(n):
            if work_adj[v]:
                has_edge = True
                break
        if not has_edge:
            break
        
        best_cycle = None
        best_len = 0
        
        # Sort vertices by degree to prioritize high-degree starts
        vertices = sorted(range(n), key=lambda v: len(work_adj[v]), reverse=True)
        
        # Try a few starts and strategies
        starts_to_try = vertices[:min(10, len(vertices))]
        
        for start in starts_to_try:
            if time.time() > deadline:
                break
            if len(work_adj[start]) < 2:
                continue
                
            for strat in strategies:
                if time.time() > deadline:
                    break
                # Use a deterministic seed based on state to allow reproducibility within a run,
                # but different across restarts via seed_offset
                rng = random.Random((start * 100 + strat + seed_offset + len(remaining_edges)) % 100000)
                cycle = _find_cycle_greedy(work_adj, n, start, deadline, rng, strat)
                if cycle and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= n: # Hamiltonian
                        break
                    
        if best_cycle:
            # Remove edges of the cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                work_adj[u].discard(v)
                work_adj[v].discard(u)
                remaining_edges.discard((min(u, v), max(u, v)))
            pieces.append(best_cycle)
        else:
            # No more cycles found by the greedy heuristic
            # Break out and treat remaining edges as single edges
            break
            
    # Add remaining edges as single edges
    for v in range(n):
        for w in work_adj[v]:
            if v < w:
                pieces.append([v, w])
                
    return pieces

def _solve_one(n, p, seed, time_budget):
    edges = _instance(n, p, seed)
    adj = _adj_from_edges(edges, n)
    m = len(edges)
    
    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in edges]
    best_count = m
    
    deadline = _T0 + _DEADLINE - 2 # Safety margin
    
    # Run multiple restarts of the decomposition with different random seeds
    # to explore different cycle packings
    num_restarts = 50 # We have ~40s per graph, this should be fast enough
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        pieces = _decompose_graph(n, edges, deadline, seed_offset=i)
        
        if _validate(adj, n, pieces):
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces

    return best_pieces

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time per graph
    # Total deadline 145s.
    # Graph 1: 20 nodes. Graph 2: 25 nodes. Graph 3: 30 nodes.
    # Larger graphs take longer.
    # Let's give 40s, 40s, 40s and keep 25s buffer?
    # Or just check deadline in _solve_one.
    
    for spec in specs:
        if time.time() - _T0 > _DEADLINE - 5:
            # Fallback: all single edges
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([[u, v] for (u, v) in edges])
            continue
        try:
            pieces = _solve_one(spec[0], spec[1], spec[2], time_budget=40.0)
        except Exception:
            edges = _instance(spec[0], spec[1], spec[2])
            pieces = [[u, v] for (u, v) in edges]
        results.append(pieces)
    return results
