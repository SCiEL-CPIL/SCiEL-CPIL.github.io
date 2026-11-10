"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle_from_start(adj, start, remaining, max_depth=20):
    """Find the longest cycle starting and ending at start using randomized DFS."""
    best_cycle = None
    best_len = 0
    
    # Randomized DFS to find long cycles
    for _ in range(50):  # Try 50 random paths
        stack = [(start, [start], {start})]
        found = False
        while stack and not found:
            node, path, visited = stack.pop()
            
            if len(path) > 1:
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path
                        cycle_edges = set()
                        valid = True
                        for i in range(len(cycle)):
                            a, b = cycle[i], cycle[(i+1) % len(cycle)]
                            e = tuple(sorted((a, b)))
                            if e in remaining:
                                cycle_edges.add(e)
                            else:
                                valid = False
                                break
                        if valid and len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = (cycle, cycle_edges)
                        found = True
                        break
                    elif nb not in visited and len(path) < max_depth:
                        stack.append((nb, path + [nb], visited | {nb}))
        
        if best_len >= 15:  # Good enough, stop searching
            break
    
    return best_cycle

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges using randomized greedy approach."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = set(edges_set)
    
    while remaining and time.time() < deadline:
        # Find all vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree (descending) to prioritize high-degree vertices
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try multiple random starts
        for start in candidates[:30]:
            if len(adj[start]) < 2:
                continue
            
            cycle = _find_longest_cycle_from_start(adj, start, remaining)
            if cycle:
                c, ce = cycle
                if len(c) > best_len:
                    best_len = len(c)
                    best_cycle = cycle
            
            if best_len >= 12:  # Found a long enough cycle
                break
        
        if best_cycle:
            cycle, cycle_edges = best_cycle
            pieces.append(cycle)
            for e in cycle_edges:
                remaining.discard(e)
            for u, v in cycle_edges:
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break
    
    # Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time per graph
    total_deadline = time.time() + 110.0  # Leave 10s margin
    time_per_graph = 35.0
    
    for i, spec in enumerate(specs):
        if time.time() > total_deadline - 5:
            # Not enough time, return trivial decomposition
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        graph_deadline = min(total_deadline - 5, time.time() + time_per_graph)
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
