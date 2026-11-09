# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge set
    adj = defaultdict(set)
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        
    # Convert to list of tuples for faster iteration if needed, but sets are good
    # We need to find up to 3 edge-disjoint Hamiltonian cycles.
    
    def find_hamiltonian_cycle(used_edges):
        """
        Finds a Hamiltonian cycle in the graph defined by 'adj' 
        avoiding edges in 'used_edges'.
        Returns a list of vertices or None.
        """
        # Use backtracking with heuristic ordering
        # Start node fixed to 0 for determinism in search space, or random?
        # Random start can help escape local minima, but 0 is fine if we shuffle neighbors.
        
        best_cycle = None
        
        # Try multiple random orderings of neighbors
        for attempt in range(100):
            random.shuffle(list(adj[0])) # This doesn't work directly on sets, need to copy
            
            # We will use a recursive backtracking function
            path = [0]
            visited = {0}
            
            def backtrack():
                if len(path) == n:
                    last = path[-1]
                    first = path[0]
                    if (min(last, first), max(last, first)) in edge_set and (min(last, first), max(last, first)) not in used_edges:
                        return True
                    return False
                
                current = path[-1]
                # Get neighbors not visited and not used
                neighbors = [v for v in adj[current] if v not in visited]
                
                # Heuristic: sort by degree in remaining graph (fewest options first)
                # This is expensive to compute fully, so we use a simple proxy:
                # Sort by number of unvisited neighbors
                neighbors.sort(key=lambda v: len([x for x in adj[v] if x not in visited]))
                
                for v in neighbors:
                    e = (min(current, v), max(current, v))
                    if e in used_edges:
                        continue
                    path.append(v)
                    visited.add(v)
                    if backtrack():
                        return True
                    path.pop()
                    visited.remove(v)
                
                return False
            
            if backtrack():
                return path[:]
                
        return None

    used_edges = set()
    cycles = []
    
    # Try to find 3 cycles
    # We run multiple trials with different random seeds to ensure we find a solution
    for trial in range(20):
        random.seed(trial)
        current_used = set()
        current_cycles = []
        
        for _ in range(3): # Max 3 possible
            cycle = find_hamiltonian_cycle(current_used)
            if cycle is None:
                break
            
            # Verify cycle edges are disjoint
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in current_used:
                    valid = False
                    break
                current_used.add(e)
            
            if valid:
                current_cycles.append(cycle)
            else:
                break
        
        if len(current_cycles) > len(cycles):
            cycles = current_cycles[:]
            
        if len(cycles) == 3:
            break
            
    # Ensure types are correct
    result = [list(c) for c in cycles]
    return result
# EVOLVE-BLOCK-END
