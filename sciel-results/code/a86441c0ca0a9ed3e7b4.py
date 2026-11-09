"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, deadline):
    n = P['n']
    start = 0
    path = [start]
    visited = {start}
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if (path[-1], start) in adj[path[-1]] or (start, path[-1]) in adj[path[-1]]:
                # Check edge exists
                if start in adj[path[-1]]:
                    return list(path)
            return None
        last = path[-1]
        for nxt in sorted(adj[last]):
            if nxt not in visited:
                path.append(nxt)
                visited.add(nxt)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    return backtrack()

def construct():
    deadline = time.time() + 55  # safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = _build_adj(edges)
    
    # Track used edges
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for _ in range(3):
        if time.time() > deadline:
            break
        
        # Build available adjacency (remove used edges)
        avail_adj = [set() for _ in range(n)]
        for u, v in edges:
            key = _edge_key(u, v)
            if key not in used_edges:
                avail_adj[u].add(v)
                avail_adj[v].add(u)
        
        # Find a Hamiltonian cycle in the available graph
        cycle = _find_hamiltonian_cycle(avail_adj, deadline)
        
        if cycle is None:
            break
        
        # Mark edges in this cycle as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(cycle)
    
    # Ensure at least one cycle
    if not cycles:
        # Fallback: return a single Hamiltonian cycle using all edges
        cycle = _find_hamiltonian_cycle(adj, deadline)
        if cycle is not None:
            cycles = [cycle]
        else:
            cycles = [list(range(n))]
    
    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
