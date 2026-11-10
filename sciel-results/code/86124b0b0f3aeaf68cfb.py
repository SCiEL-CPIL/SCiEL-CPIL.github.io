"""Combined program for cycle decomposition task."""
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

def _find_cycle_greedy(start, adj, remaining, max_len=12, rng=None):
    """Find a cycle starting from start using randomized greedy DFS."""
    if rng is None:
        rng = random.Random()
    
    best_cycle = None
    best_len = 0
    
    # Try multiple random walks to find good cycles
    for attempt in range(20):
        visited = {start}
        path = [start]
        current = start
        
        for _ in range(max_len):
            neighbors = list(adj.get(current, []))
            if not neighbors:
                break
            
            # Shuffle for randomization
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                # Check if we can close a cycle
                if nb == start and len(path) >= 3:
                    # Verify all edges are in remaining
                    valid = True
                    for i in range(len(path)):
                        a, b = path[i], path[(i + 1) % len(path)]
                        e = tuple(sorted((a, b)))
                        if e not in remaining:
                            valid = False
                            break
                    if valid and len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path
                    break
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    current = nb
                    break
            else:
                break
        
        if best_cycle and best_len >= 6:
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
    rng = random.Random(42)
    
    # Phase 1: Find cycles greedily with randomization
    while remaining and time.time() < deadline:
        # Find vertices with degree >= 2, sorted by degree descending
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree descending
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try top candidates with randomization
        for start in candidates[:25]:
            if len(adj[start]) < 2:
                continue
            
            cycle = _find_cycle_greedy(start, adj, remaining, max_len=12, rng=rng)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            
            # Stop early if we found a long enough cycle
            if best_len >= 7:
                break
        
        if best_cycle and best_len >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            break
    
    # Phase 2: Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # Evaluator kills at 150s, leave 10s margin
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        # Allocate time proportionally: ~45s per graph
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
