"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(vertices, adj, used_edges):
    """Check if the graph induced by vertices with edges not in used_edges is connected."""
    if not vertices:
        return True
    
    vset = set(vertices)
    start = next(iter(vset))
    visited = {start}
    stack = [start]
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in vset and v not in visited:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    visited.add(v)
                    stack.append(v)
    
    return len(visited) == len(vset)

def _find_ham_cycle(vertices, adj, used_edges, deadline, rng, max_depth=5000):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking."""
    vertices = list(vertices)
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # Precompute adjacency as frozensets for faster lookup
    adj_frozen = {u: frozenset(v) for u, v in adj.items()}
    
    def dfs(current, visited, path):
        if time.time() > deadline:
            return None
            
        if len(visited) == n:
            # Check if we can close the cycle
            if start in adj_frozen[current]:
                e = (min(current, start), max(current, start))
                if e not in used_edges:
                    return path
            return None
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj_frozen[current]:
            if nb in vset and nb not in visited:
                # Check if edge (current, nb) is not used
                e = (min(current, nb), max(current, nb))
                if e not in used_edges:
                    unvisited_neighbors.append(nb)
        
        if not unvisited_neighbors:
            return None
        
        # Sort by degree in remaining graph (Warnsdorff's heuristic)
        # Add a bit of randomness to break ties
        def heuristic_key(x):
            deg = 0
            for y in adj_frozen[x]:
                if y in vset and y not in visited and (min(x, y), max(x, y)) not in used_edges:
                    deg += 1
            return (deg, rng.random())
            
        unvisited_neighbors.sort(key=heuristic_key)
        
        for nb in unvisited_neighbors:
            visited.add(nb)
            path.append(nb)
            e = (min(current, nb), max(current, nb))
            used_edges.add(e)
            
            result = dfs(nb, visited, path)
            if result is not None:
                return result
            
            # Backtrack
            path.pop()
            visited.discard(nb)
            used_edges.discard(e)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(start, visited, path)

def construct():
    # The graph is C_21(1, 2, 5).
    # We can explicitly construct 3 edge-disjoint Hamiltonian cycles.
    # Cycle 1: Step 1
    cycle1 = list(range(21))
    
    # Cycle 2: Step 2
    cycle2 = []
    v = 0
    for _ in range(21):
        cycle2.append(v)
        v = (v + 2) % 21
        
    # Cycle 3: Step 5
    cycle3 = []
    v = 0
    for _ in range(21):
        cycle3.append(v)
        v = (v + 5) % 21
        
    return [cycle1, cycle2, cycle3]
# EVOLVE-BLOCK-END
