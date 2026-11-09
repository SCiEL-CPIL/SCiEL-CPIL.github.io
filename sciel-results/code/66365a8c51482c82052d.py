import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        min_deg_req = n - 39
        
        # Check if we have enough time left to try this n
        if time.time() > deadline - 1:
            break
            
        # Allocate 20s per n
        time_limit_for_n = 20.0
        n_start_time = time.time()
        
        # Ensure we don't exceed global deadline
        effective_limit = min(n_start_time + time_limit_for_n, deadline)
        
        while time.time() < effective_limit:
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all possible edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            # Greedy addition
            for u, v in all_edges:
                # Check C4 constraint: |N(u) ∩ N(v)| <= 1
                if len(adj[u]) < len(adj[v]):
                    common = 0
                    for w in adj[u]:
                        if w in adj[v]:
                            common += 1
                            if common > 1:
                                break
                else:
                    common = 0
                    for w in adj[v]:
                        if w in adj[u]:
                            common += 1
                            if common > 1:
                                break
                            
                if common <= 1:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    
            # Check minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                return {'n': n, 'edges': edges}
                
    # Fallback: return a valid small graph
    # n=39, min_deg=0. Empty graph works.
    return {'n': 39, 'edges': []}
