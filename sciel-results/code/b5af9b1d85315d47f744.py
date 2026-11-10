import time
import random
from copy import deepcopy

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for fast edge removal
    base_adj = [set() for _ in range(n)]
    for u, v in edges:
        base_adj[u].add(v)
        base_adj[v].add(u)

    def find_hamiltonian_cycle(adj, start, rng=None):
        """
        Finds a Hamiltonian cycle starting at 'start' in graph 'adj'.
        Uses Warnsdorff heuristic with optional randomization.
        Returns list of vertices or None.
        """
        if rng is None:
            rng = random
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
            
            # Heuristic: prefer neighbors with fewer unvisited neighbors (Warnsdorff)
            def score(v):
                unvisited_count = sum(1 for w in adj[v] if not visited[w])
                # If last step, must connect to start
                if len(path) == n - 1:
                    if start not in adj[v]:
                        return float('inf')
                # Add small random noise to break ties
                return unvisited_count + rng.random() * 0.1
            
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

    def try_decompose(base_adj_sets, seed=0):
        """
        Greedily tries to find 3 edge-disjoint HCs.
        Returns list of cycles found.
        """
        rng = random.Random(seed)
        cycles = []
        # Work with copies of the adjacency sets
        current_adj = [set(s) for s in base_adj_sets]
        
        for cycle_idx in range(3):
            # Try multiple start nodes with different random seeds
            found_cycle = None
            # Try start=0 first, then others
            starts_to_try = [0] + list(range(1, n))
            for s in starts_to_try:
                if not current_adj[s]:
                    continue
                cycle = find_hamiltonian_cycle(current_adj, s, rng)
                if cycle is not None:
                    found_cycle = cycle
                    break
            
            if found_cycle is None:
                break
                
            cycle = [int(v) for v in found_cycle]
            cycles.append(cycle)
            
            # Remove edges from current_adj
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v in current_adj[u]:
                    current_adj[u].discard(v)
                if u in current_adj[v]:
                    current_adj[v].discard(u)
                    
        return cycles

    best_cycles = []
    start_time = time.time()
    limit = 85.0  # Safety margin for 90s limit
    
    # First, try with deterministic approach (no randomness in ordering)
    # This should be fast and reliable
    cycles = try_decompose(base_adj, seed=42)
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
        if len(best_cycles) == 3:
            return best_cycles
    
    # If that didn't work, try multiple random seeds
    for seed in range(1, 10000):
        if time.time() - start_time > limit:
            break
        cycles = try_decompose(base_adj, seed=seed)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break
    
    return best_cycles
# EVOLVE-BLOCK-END
