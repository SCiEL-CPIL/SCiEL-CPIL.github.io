import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        min_deg_req = n - 39
        
        # If n=47, min_deg_req=8. If n=46, min_deg_req=7.
        # ex(n, C4) limits are tight.
        
        found = False
        # Number of trials depends on n
        max_trials = 500 if n <= 46 else 100
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all possible edges
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            # Greedy add
            for u, v in possible_edges:
                if not (adj[u] & adj[v]):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
                    
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                found = True
                best_n = n
                best_edges = edges
                break
                
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
