"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_greedy(adj, n, start, rng):
    """Find a cycle starting from start using randomized DFS."""
    visited = {start}
    path = [start]
    stack = [(start, list(adj[start]))]
    found = False
    
    while stack and not found:
        node, neighbors = stack[-1]
        rng.shuffle(neighbors)
        advanced = False
        for nxt in neighbors:
            if nxt == start and len(path) >= 3:
                found = True
                break
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                stack.append((nxt, list(adj[nxt])))
                advanced = True
                break
        if not advanced and not found:
            stack.pop()
            path.pop()
    
    if found:
        return list(path)
    return None

def _decompose(edges, n, rng=None):
    """Decompose edges into cycles and single edges using cycle packing with vertex coverage priority."""
    if rng is None:
        rng = Random(42)
    
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    # Phase 1: Find cycles greedily, prioritizing long cycles and vertex coverage
    max_iterations = 500
    iters = 0
    while remaining and iters < max_iterations:
        # Find the best cycle among several random attempts
        best_cycle = None
        best_score = 0
        
        # Number of attempts scales with remaining edges
        num_attempts = min(50, max(10, len(remaining) // 5))
        
        for attempt in range(num_attempts):
            # Pick start: prefer high-degree vertices
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            
            # Weight by degree to prefer high-degree starts
            weights = [len(adj[v]) for v in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            start = candidates[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    start = candidates[i]
                    break
            
            cycle = _find_cycle_greedy(adj, n, start, rng)
            if cycle and len(cycle) >= 3:
                # Score: prefer longer cycles
                score = len(cycle)
                if score > best_score:
                    best_score = score
                    best_cycle = list(cycle)
        
        if best_cycle is None or len(best_cycle) < 3:
            break
        
        # Verify and extract cycle
        valid = True
        cycle_edges = []
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            edge = tuple(sorted((u, v)))
            if edge not in remaining:
                valid = False
                break
            cycle_edges.append(edge)
        
        if not valid:
            break
        
        # Remove cycle edges
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(best_cycle[:])
        iters += 1
    
    # Phase 2: Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        rng = Random(42 + spec[2])
        pieces = _decompose(edges, n, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
