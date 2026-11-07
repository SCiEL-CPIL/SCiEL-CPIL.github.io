import sys
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        
    # Pre-compute all edges as frozensets or tuples for fast lookup
    all_edges = set()
    for u, v in P['edges']:
        if u > v:
            u, v = v, u
        all_edges.add((u, v))
        
    def get_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    def find_ham_cycle(used_edges):
        # Randomized DFS to find a Hamiltonian cycle
        # Start node 0
        path = [0]
        visited = {0}
        
        def dfs(current):
            if len(path) == n:
                # Check closing edge
                if get_edge(current, 0) not in used_edges:
                    return list(path)
                return None
            
            # Get available neighbors
            neighbors = []
            for nb in adj[current]:
                if nb not in visited:
                    if get_edge(current, nb) not in used_edges:
                        neighbors.append(nb)
            
            # Shuffle to randomize search order
            random.shuffle(neighbors)
            
            for nb in neighbors:
                path.append(nb)
                visited.add(nb)
                
                # Optimization: Check if the path can still be completed?
                # Simple check: Is the last vertex connected to any unvisited vertex or 0?
                # If not, backtrack immediately.
                # This is a bit expensive to check fully, but we can check degree.
                
                result = dfs(nb)
                if result is not None:
                    return result
                    
                path.pop()
                visited.remove(nb)
                
            return None
            
        return dfs(0)

    best_cycles = []
    # Set a time limit for the whole construct
    start_time = time.time()
    time_limit = 50 # Leave 10s buffer
    
    # Try multiple random restarts
    # Each restart tries to greedily pack cycles
    for i in range(500): # Adjust based on speed
        if time.time() - start_time > time_limit:
            break
            
        used_edges = set()
        current_cycles = []
        
        while True:
            # Try to find a cycle
            # To increase diversity, we could shuffle the global edge list or adj lists?
            # Shuffling neighbors in DFS is enough.
            
            # We can also try different starting vertices?
            # The problem says "same vertex set". A Hamiltonian cycle visits all.
            # So starting vertex doesn't change the set of vertices, just the rotation.
            # But it might change the specific cycle found.
            
            cycle = find_ham_cycle(used_edges)
            
            if cycle is None:
                break
            
            current_cycles.append(cycle)
            
            # Mark edges as used
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                used_edges.add(get_edge(u, v))
                
            # Max 10 cycles allowed by contract, but likely less is possible
            if len(current_cycles) >= 10:
                break
        
        if len(current_cycles) > len(best_cycles):
            best_cycles = current_cycles
            # If we found 10, we can stop? The contract says "Maximize number".
            # 10 is the cap. If we hit 10, it's likely optimal or near-optimal.
            # But maybe 10 is impossible? Let's assume we want to keep searching for more if < 10.
            # If we found 10, we can break early as it's the max allowed.
            if len(best_cycles) == 10:
                break
                
    # Ensure we return a valid non-empty list
    # If best_cycles is empty, we must find at least one.
    # The graph is connected and has cycles, so this should not happen.
    # But to be safe, if empty, do one deterministic try.
    if not best_cycles:
         # Fallback to deterministic or simple random
         used_edges = set()
         cycle = find_ham_cycle(used_edges)
         if cycle:
             best_cycles = [cycle]
             
    return best_cycles

# EVOLVE-BLOCK-END
