import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: 17-vertex Paley graph
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    # Try to find a 35-vertex graph
    n = 35
    
    # Initialize random graph
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.4:
                adj[i, j] = True
                adj[j, i] = True
                
    def is_valid(adj):
        # Check K4
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    # Common neighbors of i and j
                    common = np.where(adj[i] & adj[j])[0]
                    for k in common:
                        for l in common:
                            if k < l and adj[k, l]:
                                return False # K4 found
        # Check Indep 6
        # This is expensive. 
        # If no K4, we still need to check Indep 6.
        # We can sample? No, must be exact.
        # Iterate all 6-sets?
        from itertools import combinations
        for combo in combinations(range(n), 6):
            is_indep = True
            for u in range(6):
                for v in range(u+1, 6):
                    if adj[combo[u], combo[v]]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False
        return True

    # WalkSAT Loop
    while time.time() < deadline:
        if is_valid(adj):
            edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j]]
            return {'n': n, 'edges': edges}
            
        # Find a violation
        # 1. Try to find a K4
        violation_found = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = np.where(adj[i] & adj[j])[0]
                    for k in common:
                        for l in common:
                            if k < l and adj[k, l]:
                                # Found K4 {i, j, k, l}
                                # Flip one of the edges
                                candidates = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                                u, v = random.choice(candidates)
                                adj[u, v] = not adj[u, v]
                                adj[v, u] = adj[u, v]
                                violation_found = True
                                break
                        if violation_found: break
                    if violation_found: break
                if violation_found: break
            if violation_found: break
            
        if violation_found:
            continue
            
        # 2. Try to find an Indep 6
        from itertools import combinations
        for combo in combinations(range(n), 6):
            is_indep = True
            for u in range(6):
                for v in range(u+1, 6):
                    if adj[combo[u], combo[v]]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                # Found Indep 6
                # Add an edge
                u, v = random.choice([(combo[a], combo[b]) for a in range(6) for b in range(a+1, 6)])
                adj[u, v] = True
                adj[v, u] = True
                break
        
    # Return fallback
    return get_paley_17()
