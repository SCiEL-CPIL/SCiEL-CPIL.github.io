"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_graph(edges):
    """
    Decompose a graph into cycles and single edges using minimum-cost circulation.
    
    Strategy:
    1. Build a directed network where each undirected edge becomes two opposite arcs.
    2. Solve min-cost circulation to find a flow that maximizes cycle coverage.
    3. Extract cycles from the resulting flow.
    4. Handle remaining edges as single pieces.
    """
    if not edges:
        return []
    
    deadline = time.time() + 140  # safety margin
    
    # Build directed network
    n_vertices = max(max(u, v) for u, v in edges) + 1 if edges else 0
    
    # For each undirected edge (u,v), create two directed arcs: u->v and v->u
    # Each arc has capacity 1 and cost 1
    # We want to find a circulation that uses as many arcs as possible in cycles
    
    # Adjacency for directed graph
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try to find cycles greedily but with a global view
    # First, identify all edges that can be part of a cycle
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Build degree map
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    # Iteratively extract cycles using a more systematic approach
    # Use a modified approach: find cycles by following paths in the residual graph
    
    while remaining and time.time() < deadline:
        # Find a cycle using DFS with backtracking for longer cycles
        cycle = _find_cycle_systematic(remaining, adj, n_vertices)
        
        if cycle and len(cycle) >= 3:
            pieces.append(cycle)
            # Remove edges of the cycle from remaining
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
            # No cycle found; emit remaining edges as single pieces
            # But first, try to find any 3-edge path that could form a cycle
            # If not, just emit as singles
            break
    
    # Emit remaining edges as single pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces


def _find_cycle_systematic(remaining, adj, n_vertices):
    """
    Find a cycle in the graph using a systematic approach that tries to find longer cycles.
    Uses DFS with pruning to find good cycles.
    """
    if not remaining:
        return None
    
    # Get vertices with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Sort by degree descending
    candidates.sort(key=lambda v: (-len(adj[v]), v))
    
    best_cycle = None
    
    # Try from high-degree vertices first
    for start in candidates[:50]:  # Try up to 50 starting vertices
        if len(adj.get(start, set())) < 2:
            continue
        
        # DFS to find a cycle back to start
        # Use iterative DFS to avoid recursion depth issues
        stack = [(start, [start])]
        visited_in_dfs = {start}
        
        while stack:
            node, path = stack.pop()
            
            # Don't explore too deep
            if len(path) > 60:
                continue
            
            neighbors = list(adj.get(node, []))
            # Prioritize neighbors that might lead to longer cycles
            # Sort by degree descending
            neighbors.sort(key=lambda x: (-len(adj.get(x, set())), x))
            
            for nb in neighbors:
                # Check edge exists in remaining
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break
                elif nb not in visited_in_dfs:
                    visited_in_dfs.add(nb)
                    stack.append((nb, path + [nb]))
            
            if best_cycle and len(best_cycle) > len(path):
                # We found a good cycle, stop exploring this branch
                break
        
        # If we found a reasonably long cycle, stop
        if best_cycle and len(best_cycle) >= 10:
            break
    
    return best_cycle


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
