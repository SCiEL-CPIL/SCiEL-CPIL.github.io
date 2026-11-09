"""Improved program for the suite task."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and edge set
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
        
    def get_edge(u, v):
        return (min(u,v), max(u,v))

    def find_ham_cycle(used_edges):
        """
        Finds a Hamiltonian cycle in the graph avoiding 'used_edges'.
        Uses a randomized backtracking search with heuristics.
        """
        vertices = list(range(n))
        
        # Try multiple random permutations as starting points
        for attempt in range(100):
            random.shuffle(vertices)
            start = vertices[0]
            path = [start]
            visited = {start}
            
            def backtrack():
                if len(path) == n:
                    last = path[-1]
                    first = path[0]
                    e = get_edge(last, first)
                    if e in edge_set and e not in used_edges:
                        return True
                    return False
                
                current = path[-1]
                # Get unvisited neighbors
                neighbors = [v for v in adj[current] if v not in visited]
                
                # Heuristic: Sort neighbors by degree in remaining graph (Warnsdorff's rule)
                # Fewest options first
                def degree_in_remaining(v):
                    count = 0
                    for u in adj[v]:
                        if u not in visited:
                            count += 1
                    return count
                
                neighbors.sort(key=degree_in_remaining)
                
                for v in neighbors:
                    e = get_edge(current, v)
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
                return path
        
        return None

    # Main logic: Find 3 edge-disjoint cycles
    best_cycles = []
    start_time = time.time()
    
    for trial in range(50):
        random.seed(trial)
        used_edges = set()
        cycles = []
        
        success = True
        for _ in range(3):
            cycle = find_ham_cycle(used_edges)
            if cycle is None:
                success = False
                break
            
            # Verify and add edges
            valid = True
            for i in range(n):
                u, v = cycle[i], cycle[(i+1) % n]
                e = get_edge(u, v)
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            
            if not valid:
                success = False
                break
            
            cycles.append(cycle)
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
            if len(best_cycles) == 3:
                break
        
        if time.time() - start_time > 55:
            break
            
    # Ensure types are correct
    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
