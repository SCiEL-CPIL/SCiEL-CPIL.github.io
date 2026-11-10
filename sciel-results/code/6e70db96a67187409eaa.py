import time
import random
from copy import deepcopy

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    base_adj = [[] for _ in range(n)]
    for u, v in edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    for i in range(n):
        base_adj[i].sort()

    def find_hamiltonian_cycle(adj, start):
        """
        Finds a Hamiltonian cycle starting at 'start' in graph 'adj'.
        Uses randomized Warnsdorff heuristic.
        Returns list of vertices or None.
        """
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack(current):
            if len(path) == n:
                if start in adj[current]:
                    return True
                return False
            
            neighbors = [v for v in adj[current] if not visited[v]]
            if not neighbors:
                return False
            
            # Heuristic score
            def score(v):
                unvisited_count = sum(1 for w in adj[v] if not visited[w])
                # If last step, must connect to start
                if len(path) == n - 1:
                    if start not in adj[v]:
                        return float('inf')
                # Add small random noise to break ties and diversify search
                return unvisited_count + random.random() * 0.1
            
            neighbors.sort(key=score)
            
            for v in neighbors:
                if len(path) == n - 1 and start not in adj[v]:
                    continue
                
                path.append(v)
                visited[v] = True
                
                if backtrack(v):
                    return True
                    
                path.pop()
                visited[v] = False
                
            return False

        if backtrack(start):
            return path
        return None

    def try_decompose(adj):
        """
        Greedily tries to find 3 edge-disjoint HCs.
        Returns list of cycles found.
        """
        cycles = []
        current_adj = deepcopy(adj)
        
        for _ in range(3):
            # Try a few random starts if 0 fails? 
            # Or just try 0. 
            # If 0 fails, try other vertices.
            start = 0
            cycle = find_hamiltonian_cycle(current_adj, start)
            
            if cycle is None:
                # Try other start nodes
                found = False
                for s in range(1, n):
                    if current_adj[s]: # If s has edges
                        cycle = find_hamiltonian_cycle(current_adj, s)
                        if cycle:
                            found = True
                            break
                if not found:
                    break
            else:
                found = True
            
            if not found:
                break
                
            cycle = [int(v) for v in cycle]
            cycles.append(cycle)
            
            # Remove edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v in current_adj[u]:
                    current_adj[u].remove(v)
                if u in current_adj[v]:
                    current_adj[v].remove(u)
                    
        return cycles

    best_cycles = []
    start_time = time.time()
    limit = 80.0 # Safety margin for 90s limit
    
    # Run multiple attempts to increase chance of finding 3
    # Since the graph is small, each attempt is fast.
    while time.time() - start_time < limit:
        cycles = try_decompose(base_adj)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break # Max possible
        
        # Small sleep to prevent CPU spin if we are very fast? 
        # No, we want to try as many as possible.
        
    return best_cycles
# EVOLVE-BLOCK-END
