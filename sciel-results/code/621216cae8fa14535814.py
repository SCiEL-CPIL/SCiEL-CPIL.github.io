import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    n = P['n']
    
    # Build adjacency and edge set
    edge_set = set()
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        ek = (min(u,v), max(u,v))
        edge_set.add(ek)
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def find_hamiltonian_cycle(available_set, start, time_limit):
        """Find a Hamiltonian cycle using backtracking with degree-based pruning."""
        t0 = time.time()
        path = [start]
        visited = {start}
        
        # Precompute degree in available edges
        deg_in_avail = [0] * n
        for u, v in available_set:
            deg_in_avail[u] += 1
            deg_in_avail[v] += 1
        
        # Quick check: all vertices must have degree >= 2
        for i in range(n):
            if deg_in_avail[i] < 2:
                return None
        
        avail_set = available_set
        
        def backtrack():
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in avail_set:
                    return list(path)
                return None
            
            last = path[-1]
            candidates = []
            for nb in adj[last]:
                if nb not in visited and edge_key(last, nb) in avail_set:
                    # Check if nb would still have enough degree
                    if deg_in_avail[nb] - 1 >= 1:
                        candidates.append(nb)
            
            if not candidates:
                return None
            
            # Sort by degree (most constrained first)
            candidates.sort(key=lambda x: deg_in_avail[x])
            
            for nb in candidates:
                ek = edge_key(last, nb)
                
                # Check all unvisited vertices maintain degree >= 2 after removing ek
                valid = True
                for v in range(n):
                    if v in visited or v == nb:
                        continue
                    deg_v = deg_in_avail[v]
                    if v == last:
                        deg_v -= 1
                    if deg_v < 2:
                        valid = False
                        break
                
                if valid:
                    path.append(nb)
                    visited.add(nb)
                    result = backtrack()
                    if result is not None and result != "timeout":
                        return result
                    path.pop()
                    visited.remove(nb)
                    
            return None
        
        result = backtrack()
        if result == "timeout":
            return None
        return result
    
    cycles = []
    available = set(edge_set)
    
    # Maximum possible: floor(70/21) = 3
    max_possible = len(edge_set) // n
    
    for cycle_idx in range(max_possible):
        if time.time() > deadline:
            break
        
        found = None
        remaining_time = deadline - time.time()
        
        # Try multiple starting vertices with decreasing time limits
        starts = list(range(n))
        random.shuffle(starts)
        
        for start in starts:
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 0.3:
                break
            time_limit = min(remaining_time * 0.3, 10.0)
            found = find_hamiltonian_cycle(available, start, time_limit)
            if found is not None:
                break
        
        # If not found, try with different random seeds
        if found is None:
            for seed in range(5):
                if time.time() > deadline:
                    break
                random.seed(seed + cycle_idx * 100)
                remaining_time = deadline - time.time()
                if remaining_time < 0.3:
                    break
                start = random.randint(0, n-1)
                time_limit = min(remaining_time * 0.2, 5.0)
                found = find_hamiltonian_cycle(available, start, time_limit)
                if found is not None:
                    break
        
        if found is None:
            break
        
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i+1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
        
        # Check if remaining graph can still support a Hamiltonian cycle
        if len(available) < n:
            break
        # Quick degree check
        deg_check = [0] * n
        for u, v in available:
            deg_check[u] += 1
            deg_check[v] += 1
        if any(d < 2 for d in deg_check):
            break
    
    if cycles:
        return [list(c) for c in cycles]
    else:
        # Fallback: try to find at least 1 cycle
        for start in range(n):
            found = find_hamiltonian_cycle(set(edge_set), start, 2.0)
            if found:
                return [found]
        return [list(range(n))]
# EVOLVE-BLOCK-END
