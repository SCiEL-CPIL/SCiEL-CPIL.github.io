"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(start, adj, remaining, max_len=12):
    """Find a long cycle starting from start using iterative deepening DFS."""
    best_cycle = None
    best_len = 0
    
    # Use DFS with path tracking, prefer longer cycles
    # We'll try to find cycles of increasing length
    for target_len in range(max_len, 2, -1):
        if best_len >= target_len:
            break
        # Try to find a cycle of exactly target_len
        path = [start]
        visited = {start}
        
        def dfs(node, depth):
            nonlocal best_cycle, best_len
            if depth == target_len - 1:
                # Check if we can close the cycle
                if start in adj[node] and (tuple(sorted((node, start)))) in remaining:
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    return True
                return False
            
            for nb in adj[node]:
                if nb == start:
                    continue
                if nb in visited:
                    continue
                # Check edge exists in remaining
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                visited.add(nb)
                path.append(nb)
                if dfs(nb, depth + 1):
                    path.pop()
                    visited.discard(nb)
                    return True
                path.pop()
                visited.discard(nb)
        
        if dfs(start, 1):
            if best_len >= target_len:
                break
    
    return best_cycle

def _find_any_cycle(start, adj, remaining):
    """Quickly find any cycle starting from start."""
    visited = {start}
    stack = [(start, [start])]
    
    while stack:
        node, path = stack.pop()
        for nb in adj[node]:
            e = tuple(sorted((node, nb)))
            if e not in remaining:
                continue
            if nb == start and len(path) >= 3:
                return path
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))
    
    return None

def _decompose_graph(edges, deadline):
    """Decompose edges into cycles and single edges efficiently."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Find cycles greedily, preferring longer ones
    while remaining and time.time() < deadline:
        # Find vertices with degree >= 2, sorted by degree descending
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        # Sort by degree descending to prefer high-degree vertices
        candidates.sort(key=lambda v: -len(adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try each candidate start vertex, looking for longer cycles
        for start in candidates[:40]:
            if len(adj[start]) < 2:
                continue
            
            # First try to find a long cycle
            cycle = _find_longest_cycle(start, adj, remaining, max_len=12)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            
            # If we found a reasonably long cycle, stop searching
            if best_len >= 8:
                break
        
        # If no long cycle found, try to find any cycle
        if not best_cycle:
            for start in candidates[:50]:
                if len(adj[start]) < 2:
                    continue
                cycle = _find_any_cycle(start, adj, remaining)
                if cycle and len(cycle) >= 3:
                    best_cycle = cycle
                    best_len = len(cycle)
                    break
        
        if best_cycle and best_len >= 3:
            pieces.append(best_cycle)
            # Remove cycle edges from remaining and adj
            for i in range(len(best_cycle)):
                a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
        else:
            break
    
    # Phase 2: Try to find more cycles in remaining edges (second pass)
    if remaining and time.time() < deadline:
        # Rebuild adjacency for remaining
        adj2 = defaultdict(set)
        for u, v in remaining:
            adj2[u].add(v)
            adj2[v].add(u)
        
        while remaining and time.time() < deadline:
            candidates = [v for v in adj2 if len(adj2[v]) >= 2]
            if not candidates:
                break
            
            candidates.sort(key=lambda v: -len(adj2[v]))
            
            found = False
            for start in candidates[:30]:
                if len(adj2[start]) < 2:
                    continue
                
                cycle = _find_longest_cycle(start, adj2, remaining, max_len=10)
                if cycle and len(cycle) >= 3:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((a, b)))
                        remaining.discard(e)
                        adj2[a].discard(b)
                        adj2[b].discard(a)
                    found = True
                    break
            
            if not found:
                break
    
    # Phase 3: Add remaining edges as single-edge pieces
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
        graph_deadline = deadline - (len(specs) - 1 - i) * 5.0  # Leave 5s per remaining graph
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
