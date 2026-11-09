"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import copy
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def is_edge(u, v):
        return (min(u,v), max(u,v)) in edge_set
    
    # Find a Hamiltonian cycle on a given vertex set using heuristic
    def find_hamiltonian_cycle(vertices, used_edges):
        """Find a Hamiltonian cycle on the given vertex set avoiding used_edges."""
        m = len(vertices)
        if m < 3:
            return None
        
        vset = set(vertices)
        
        # Try multiple random starts
        best_cycle = None
        for attempt in range(50):
            random.shuffle(vertices)
            start = vertices[0]
            path = [start]
            visited = {start}
            
            # Greedy with backtracking (limited depth)
            def backtrack(path, visited):
                if len(path) == m:
                    if is_edge(path[-1], path[0]) and (min(path[-1], path[0]), max(path[-1], path[0])) not in used_edges:
                        return True
                    return False
                
                current = path[-1]
                candidates = [v for v in adj[current] if v in vset and v not in visited]
                
                # Sort by degree (fewest options first)
                candidates.sort(key=lambda v: len([x for x in adj[v] if x in vset and x not in visited]))
                
                for v in candidates:
                    if (min(current, v), max(current, v)) in used_edges:
                        continue
                    path.append(v)
                    visited.add(v)
                    if backtrack(path, visited):
                        return True
                    path.pop()
                    visited.remove(v)
                
                return False
            
            if backtrack(path, visited):
                return path
        
        return None
    
    # Try to find edge-disjoint Hamiltonian cycles on the full vertex set
    all_vertices = list(range(n))
    used_edges = set()
    cycles = []
    
    # Try multiple times with different random seeds
    for trial in range(100):
        random.seed(trial)
        used_edges = set()
        cycles = []
        
        for _ in range(10):  # Try to find up to 10 cycles
            cycle = find_hamiltonian_cycle(all_vertices, used_edges)
            if cycle is None:
                break
            # Add edges to used
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                edge = (min(u,v), max(u,v))
                if edge in used_edges:
                    break
                used_edges.add(edge)
            else:
                cycles.append(cycle)
        
        if len(cycles) > len(cycles):
            pass  # Already tracking
    
    # Use the best result
    best_cycles = []
    for trial in range(200):
        random.seed(trial)
        used_edges = set()
        cycles = []
        
        for _ in range(15):
            cycle = find_hamiltonian_cycle(all_vertices, used_edges)
            if cycle is None:
                break
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                edge = (min(u,v), max(u,v))
                if edge in used_edges:
                    valid = False
                    break
                used_edges.add(edge)
            if valid:
                cycles.append(cycle)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
        
        if len(best_cycles) >= 10:
            break
    
    # Ensure types are correct
    result = [list(c) for c in best_cycles]
    return result
# EVOLVE-BLOCK-END
