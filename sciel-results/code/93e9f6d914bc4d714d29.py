"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(n, adj, start, rng, deadline):
    """Find a Hamiltonian cycle starting at 'start' using Warnsdorff's heuristic."""
    path = [start]
    visited = {start}
    
    def dfs():
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if last vertex connects back to start
            if start in adj[path[-1]]:
                return True
            return False
        
        cur = path[-1]
        # Get unvisited neighbors, sorted by Warnsdorff's rule (fewest unvisited neighbors first)
        neighbors = [v for v in adj[cur] if v not in visited]
        if not neighbors:
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (ascending)
        neighbors.sort(key=lambda v: len([w for w in adj[v] if w not in visited]))
        
        for v in neighbors:
            if time.time() > deadline:
                return False
            path.append(v)
            visited.add(v)
            if dfs():
                return True
            path.pop()
            visited.discard(v)
        
        return False
    
    if dfs():
        return list(path)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    n = P['n']
    base_adj = _build_adj(n, P['edges'])
    
    rng = random.Random(42)
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    # Each Hamiltonian cycle uses exactly n=21 edges
    # Total edges = 70, so max 3 cycles (63 edges)
    
    # Work with a mutable copy of adjacency
    adj = [set(s) for s in base_adj]
    
    best_cycles = []
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Reset adjacency for this restart attempt
        adj = [set(s) for s in base_adj]
        cycles = []
        
        # Try to find Hamiltonian cycles greedily
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Find a starting vertex
            # Prefer vertices with highest degree in current adj
            start = max(range(n), key=lambda v: len(adj[v]))
            
            ham = _find_ham_cycle(n, adj, start, rng, deadline)
            if ham is None:
                # Try different starting vertices
                found = False
                for s in range(n):
                    if time.time() > deadline:
                        break
                    ham = _find_ham_cycle(n, adj, s, rng, deadline)
                    if ham is not None:
                        found = True
                        break
                if not found:
                    break
            
            # Extract edges from Hamiltonian cycle and remove them
            for i in range(len(ham)):
                u = ham[i]
                v = ham[(i + 1) % len(ham)]
                adj[u].discard(v)
                adj[v].discard(u)
            
            cycles.append(ham)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                break
    
    # Ensure we return at least 1 cycle
    if not best_cycles:
        # Fallback: try to find any cycle
        best_cycles = [list(range(n))]
    
    # Convert to lists of Python ints
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
