import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and edge set
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def edge_key(u, v):
        return (min(u, v), max(u, v))

    # Precompute all edges as a set for fast lookup
    all_edges_set = set(edge_key(u, v) for u, v in edges)

    def find_hamiltonian_cycle(available_edges, start=0, time_limit=None):
        """
        Finds a Hamiltonian cycle using backtracking with Warnsdorff's heuristic.
        available_edges: set of edges that can be used
        start: starting vertex
        time_limit: optional time limit in seconds
        """
        if time_limit is None:
            time_limit = 0.1
            
        path = [start]
        visited = {start}
        start_time = time.time()
        
        # Pre-compute neighbors for each vertex for faster access
        # We use the global adj but filter by available_edges
        
        def backtrack():
            if time.time() - start_time > time_limit:
                return None
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            
            last = path[-1]
            
            # Get valid neighbors
            candidates = []
            for nb in adj[last]:
                if nb not in visited and edge_key(last, nb) in available_edges:
                    candidates.append(nb)
            
            if not candidates:
                return None
                
            # Warnsdorff's heuristic: sort by number of available neighbors (degree in remaining graph)
            # We need to count how many unvisited neighbors each candidate has in available_edges
            def get_degree(v):
                count = 0
                for nb in adj[v]:
                    if nb not in visited and edge_key(v, nb) in available_edges:
                        count += 1
                return count
            
            candidates.sort(key=get_degree)
            
            for nb in candidates:
                # Tentative move
                path.append(nb)
                visited.add(nb)
                
                result = backtrack()
                if result:
                    return result
                
                path.pop()
                visited.remove(nb)
                
            return None
            
        return backtrack()

    cycles = []
    available = set(all_edges_set)
    
    # We need to find as many cycles as possible (up to 10, but realistically 3 given 70 edges / 21 edges per cycle)
    # Max possible is floor(70/21) = 3
    
    for i in range(10):
        if time.time() > deadline:
            break
            
        found = None
        # Try different starts if the first one fails
        # To improve robustness, we can try a few random starts or all starts
        # Given the symmetry and density, start 0 usually works, but let's try a few.
        
        # Strategy: Try start 0 first. If it fails, try others.
        # To speed up, we can also randomize the order of candidates in Warnsdorff if it's deterministic and failing?
        # But Warnsdorff is usually good.
        
        # Let's try all starts to be safe, but break early if found.
        # Optimization: Only try starts if previous ones failed.
        
        # Actually, let's just try start 0. If it fails, try 1, 2, etc.
        # But to avoid long failures, we can add a small time limit per start attempt.
        
        for start in range(n):
            if time.time() > deadline:
                break
                
            # Give each start attempt a small time slice
            # If we have plenty of time, give more.
            # Simple heuristic: 0.5s per start attempt, but total deadline is hard.
            found = find_hamiltonian_cycle(available, start, time_limit=0.5)
            if found:
                break
        
        if not found:
            break
            
        # Remove edges of this cycle
        for j in range(len(found)):
            u = found[j]
            v = found[(j + 1) % len(found)]
            available.discard(edge_key(u, v))
            
        cycles.append(found)
        
        # If we have fewer edges than needed for a cycle, stop
        if len(available) < n:
            break

    if len(cycles) > 0:
        # Ensure types are correct
        return [list(c) for c in cycles]
    else:
        return []

# EVOLVE-BLOCK-END
