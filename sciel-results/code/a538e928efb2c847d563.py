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
    
    def dfs(current, visited, path):
        if time.time() > deadline:
            return None
            
        if len(visited) == n:
            # Check if we can close the cycle
            if start in adj[current]:
                e = (min(current, start), max(current, start))
                if e not in used_edges:
                    return path
            return None
        
        # Get unvisited neighbors
        unvisited_neighbors = []
        for nb in adj[current]:
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
            for y in adj[x]:
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
    start_time = time.time()
    total_deadline = start_time + 85  # 85 seconds to be safe
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    all_vertices = list(range(n))
    max_possible_cycles = 3 # Theoretical max for 21 vertices, 70 edges
    
    best_cycles = []
    
    # Try multiple random restarts to ensure we find the maximum set
    # Since the graph is small, we can afford a few restarts
    for attempt in range(10):
        if time.time() > total_deadline:
            break
            
        used_edges = set()
        cycles = []
        
        for i in range(max_possible_cycles):
            if time.time() > total_deadline:
                break
                
            # Check connectivity before expensive search
            if not _is_connected(all_vertices, adj, used_edges):
                break
                
            # Use a unique RNG for each attempt to vary the search
            rng = random.Random(attempt * 1000 + i)
            
            # Allocate time for this specific cycle search
            # If we have 3 cycles to find, we can give more time to earlier ones?
            # Or just use the global deadline.
            cycle = _find_ham_cycle(all_vertices, adj, used_edges, total_deadline, rng)
            
            if cycle is None:
                break
            
            # Add edges of this cycle to used_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            
            cycles.append(cycle)
            
        # Update best if we found more
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == max_possible_cycles:
                break # Found the theoretical max, stop searching

    # Convert to proper types
    result = [list(c) for c in best_cycles]
    
    # If no cycles found, return an empty list? 
    # The task says "nonempty list". 
    # But if we can't find any, returning [] might be invalid.
    # However, the seed returns 1 cycle. It is very likely we find at least 1.
    # To be safe, if best_cycles is empty, we should probably try a very simple fallback 
    # or just return [] and hope it's not penalized as "invalid" vs "low score".
    # Actually, "Any broken rule makes the whole result invalid".
    # "construct() -> nonempty list".
    # So we MUST return a non-empty list.
    # If our search fails, we should fall back to a known valid cycle.
    # The seed program returned [list(range(21))]. Let's check if that's a valid cycle.
    # Edges: (0,1), (1,2), ..., (20,0).
    # Does (20,0) exist? Yes.
    # Does (i, i+1) exist for all i?
    # 0-1: Yes
    # 1-2: Yes
    # 2-3: Yes
    # 3-4: Yes
    # 4-5: Yes
    # 5-6: Yes
    # 6-7: Yes
    # 7-8: Yes
    # 8-9: Yes
    # 9-10: Yes
    # 10-11: Yes
    # 11-12: Yes
    # 12-13: Yes
    # 13-14: Yes
    # 14-15: Yes
    # 15-16: Yes
    # 16-17: Yes
    # 17-18: Yes
    # 18-19: Yes
    # 19-20: Yes
    # 20-0: Yes
    # So [0..20] is a valid Hamiltonian cycle.
    
    if not result:
        result = [list(range(n))]

    return result
# EVOLVE-BLOCK-END
