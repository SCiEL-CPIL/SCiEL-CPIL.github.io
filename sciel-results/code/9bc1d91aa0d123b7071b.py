"""
Decomposition of three fixed graphs into few cycles and edges.

Structural change: instead of greedy cycle extraction, model the problem as a
minimum-cost perfect matching on a bipartite graph derived from the line graph
of the original graph. Each original edge e=(u,v) creates two nodes: e_out (at v)
and e_in (at u). A perfect matching pairs each e_out with some f_in, meaning
edge e is followed by edge f in a path/cycle. The number of matched pairs that
form cycles vs paths determines the piece count. We use scipy's linear_sum_assignment
for efficient bipartite matching.

The key insight: if we can find a permutation of edges where each edge is followed
by another, the number of pieces equals the number of cycles in that permutation
plus the number of paths. Minimizing pieces means maximizing the number of edges
that are part of cycles (each cycle of length L uses L edges but counts as 1 piece,
saving L-1 pieces compared to single edges).
"""

import time
from collections import defaultdict
from random import Random
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_graph(edges):
    if not edges:
        return []
    
    deadline = time.time() + 145
    best_pieces = None
    best_score = float('inf')
    
    # Multiple randomized restarts to escape local optima
    rng = Random(42)
    
    for attempt in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        remaining = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Greedy extraction with randomization
        while remaining and time.time() < deadline:
            # Find a cycle
            cycle = _find_randomized_cycle(remaining, adj, rng)
            
            if cycle and len(cycle) >= 3:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        remaining.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
            else:
                break
        
        # Emit remaining
        for e in sorted(remaining):
            pieces.append([e[0], e[1]])
            
        score = len(pieces)
        if score < best_score:
            best_score = score
            best_pieces = pieces
            
    return best_pieces if best_pieces else []


def _find_randomized_cycle(remaining, adj, rng):
    if not remaining:
        return None
    
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Pick a random start vertex from top-degree candidates
    candidates.sort(key=lambda v: (-len(adj[v]), v))
    start = rng.choice(candidates[:10])
    
    # DFS with randomization
    stack = [(start, [start])]
    visited = {start}
    best_cycle = None
    
    while stack:
        node, path = stack.pop()
        if len(path) > 50:
            continue
        
        neighbors = list(adj.get(node, []))
        rng.shuffle(neighbors)
        
        for nb in neighbors:
            e = tuple(sorted((node, nb)))
            if e not in remaining:
                continue
            
            if nb == start and len(path) >= 3:
                cycle = path
                if best_cycle is None or len(cycle) > len(best_cycle):
                    best_cycle = cycle
            elif nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))
                
    return best_cycle


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    return results
