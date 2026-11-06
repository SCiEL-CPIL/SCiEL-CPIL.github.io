import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 256
    # Vertices 2..256 mapped to 0..254
    weights = [1.0 / (i + 2) for i in range(255)]
    
    # Precompute adjacency
    adj = [set() for _ in range(255)]
    for i in range(255):
        vi = i + 2
        for j in range(i + 1, 255):
            vj = j + 2
            if lcm(vi, vj) <= 256:
                adj[i].add(j)
                adj[j].add(i)
                
    # Greedy initialization
    def greedy():
        candidates = set(range(255))
        selected = []
        s = 0.0
        while candidates:
            # Pick min index (highest weight)
            v = min(candidates)
            selected.append(v + 2)
            s += weights[v]
            # Remove v and neighbors
            candidates -= adj[v]
            candidates.discard(v)
        return s, selected

    best_sum, best_set = greedy()
    
    def search(candidates, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Upper bound
        # Sum of weights of all candidates
        # We can cache this or compute it.
        # Computing sum is O(K) where K is len(candidates)
        ub = current_sum + sum(weights[i] for i in candidates)
        if ub <= best_sum:
            return
            
        # Heuristic: pick vertex with highest weight (min index)
        v = min(candidates)
        
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_candidates = candidates - adj[v] - {v}
        search(new_candidates, current_sum + weights[v], current_set + [v + 2])
        
        # Branch 2: Exclude v
        new_candidates = candidates - {v}
        search(new_candidates, current_sum, current_set)
        
    initial_candidates = set(range(255))
    search(initial_candidates, 0.0, [])
    
    return best_set
