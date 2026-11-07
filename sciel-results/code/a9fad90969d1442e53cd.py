"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify the graph is 6-regular
    # Each vertex should have degree 6
    # Total edges = 63, so 63*2/21 = 6, yes it's 6-regular
    
    # We want to find edge-disjoint Hamiltonian cycles on all 21 vertices
    # Maximum possible: 63 edges / 21 edges per cycle = 3 cycles
    # So we're looking for up to 3 edge-disjoint Hamiltonian cycles
    
    # Strategy: Use backtracking to find edge-disjoint Hamiltonian cycles
    # First find one Hamiltonian cycle, remove its edges, repeat
    
    def find_hamiltonian_cycle(available_edges, used_edges):
        """Find a Hamiltonian cycle using only available edges not in used_edges."""
        # Build adjacency from available edges
        adj_avail = {v: set() for v in range(n)}
        for u, v in available_edges:
            if (u, v) not in used_edges and (v, u) not in used_edges:
                adj_avail[u].add(v)
                adj_avail[v].add(u)
        
        # Check if graph is still connected and each vertex has degree >= 2
        for v in range(n):
            if len(adj_avail[v]) < 2:
                return None
        
        # Use DFS/backtracking to find Hamiltonian cycle
        # Fix vertex 0 as start to avoid symmetry
        start = 0
        
        def dfs(path, visited):
            if len(path) == n:
                # Check if last vertex connects to start
                if start in adj_avail[path[-1]]:
                    return path
                return None
            
            last = path[-1]
            # Try neighbors in some order
            # Heuristic: try vertices with fewer available neighbors first (most constrained)
            neighbors = sorted(adj_avail[last], key=lambda x: len(adj_avail[x]))
            
            for nb in neighbors:
                if nb == start and len(path) < n - 1:
                    continue
                if nb not in visited:
                    result = dfs(path + [nb], visited | {nb})
                    if result is not None:
                        return result
            return None
        
        result = dfs([start], {start})
        if result is not None:
            return result
        return None
    
    def extract_cycles():
        """Find edge-disjoint Hamiltonian cycles."""
        cycles = []
        used_edges = set()
        
        for _ in range(3):  # Max 3 cycles
            # Find a Hamiltonian cycle in remaining edges
            cycle = find_hamiltonian_cycle(edges, used_edges)
            if cycle is None:
                break
            
            # Extract edges from cycle
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                cycle_edges.add(edge)
                used_edges.add(edge)
            
            cycles.append(cycle)
        
        return cycles
    
    # Try multiple random restarts with different edge orderings
    best_cycles = []
    
    # First try deterministic approach
    cycles = extract_cycles()
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    # Try with different random seeds for tie-breaking
    for seed in range(50):
        random.seed(seed)
        # Shuffle edges to break symmetry
        shuffled_edges = edges[:]
        random.shuffle(shuffled_edges)
        
        # Temporarily replace edges
        original_edges = P['edges']
        P['edges'] = shuffled_edges
        
        cycles = extract_cycles()
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        
        P['edges'] = original_edges
    
    # Convert to proper format: list of lists of ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    if not result:
        # Fallback: return a single valid cycle if we found any
        # The seed returns list(range(n)) which is a valid Hamiltonian cycle
        return [list(range(n))]
    
    return result

# EVOLVE-BLOCK-END
