import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
total_edges = len(edge_order)

def find_hamiltonian_cycle(adj, n, deadline):
    """
    Find a Hamiltonian cycle in the graph using backtracking.
    adj is a list of sets of neighbors.
    Returns a list of vertices forming the cycle, or None.
    """
    if time.time() > deadline:
        return None
    
    # Pre-check: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    
    # Fix vertex 0 as start. Also fix the second vertex to be the smallest neighbor of 0
    # to break reflection symmetry.
    neighbors_of_0 = sorted(adj[0])
    if not neighbors_of_0:
        return None
    
    best_cycle = None
    
    for second in neighbors_of_0:
        if time.time() > deadline:
            return None
        
        path = [0, second]
        used = {0, second}
        # Track which edges are used in the path (for the cycle, we need to close back to 0)
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check if we can close the cycle
                if 0 in adj[path[-1]]:
                    return list(path)
                return None
            
            current = path[-1]
            # Get unvisited neighbors, sorted by degree (Warnsdorff's heuristic)
            unvisited_neighbors = [v for v in adj[current] if v not in used]
            
            # Heuristic: sort by number of unvisited neighbors (fewest first)
            unvisited_neighbors.sort(key=lambda v: len([u for u in adj[v] if u not in used]))
            
            for next_v in unvisited_neighbors:
                # Pruning: if next_v is the only remaining unvisited vertex, 
                # it must connect back to 0
                remaining_unvisited = [v for v in range(n) if v not in used]
                if len(remaining_unvisited) == 2:  # current + next_v
                    if 0 not in adj[next_v]:
                        continue
                
                path.append(next_v)
                used.add(next_v)
                
                result = backtrack()
                if result is not None:
                    return result
                
                path.pop()
                used.remove(next_v)
            
            return None
        
        result = backtrack()
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    best_cycle = None
    best_k = total_edges + 1
    
    # Try increasing k from 24 upward
    for k in range(24, total_edges + 1):
        if time.time() > deadline - 5:
            break
        
        # Build adjacency from first k edges
        adj = [set() for _ in range(n)]
        for i in range(k):
            u, v = edge_order[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Quick check: minimum degree
        min_deg = min(len(adj[v]) for v in range(n))
        if min_deg < 2:
            continue
        
        # Try to find Hamiltonian cycle
        cycle = find_hamiltonian_cycle(adj, n, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_k = k
            break
    
    if best_cycle is None:
        # Fallback: use the identity permutation
        return list(range(n))
    
    return best_cycle
# EVOLVE-BLOCK-END
