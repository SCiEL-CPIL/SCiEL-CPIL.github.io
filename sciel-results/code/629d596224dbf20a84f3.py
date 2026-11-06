import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 39
    best_edges = []
    
    # Try n from 46 down to 39
    for n in range(46, 38, -1):
        if time.time() > deadline:
            break
            
        target_delta = n - 39
        
        # Try to find a C4-free graph on n vertices with min degree >= target_delta
        # We aim for exactly target_delta regular if possible, or higher.
        # Random regular graph generation is tricky for arbitrary d.
        # We can use a greedy approach or swap-based repair.
        
        found = False
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate a random graph with degrees close to target_delta
            # Simple method: start with empty, add random edges until degrees satisfied, then fix.
            # Better: Use a random 2-regular graph (union of cycles) and merge?
            # Or just random edges and check.
            
            # Let's use a random permutation based approach for regular graphs if n*d is even.
            # For n=46, d=7. 46*7 = 322 (even).
            # For n=45, d=6. 45*6 = 270 (even).
            
            if (n * target_delta) % 2 != 0:
                # Can't be regular. Try target_delta + 1?
                # But min degree must be >= target_delta.
                # If we can't make it regular, we can make it semi-regular.
                # For simplicity, skip if odd, or try d=target_delta+1 if possible.
                if (n * (target_delta + 1)) % 2 == 0:
                    target_delta = target_delta + 1
                else:
                    continue

            # Generate random regular graph
            # Algorithm:
            # 1. Create list of vertices with required degrees.
            # 2. Pair them up randomly.
            # This is the "Configuration Model".
            # It may produce multi-edges or self-loops. We need a simple graph.
            # We can retry if multi-edges appear.
            
            deg_list = [target_delta] * n
            ports = []
            for i in range(n):
                for _ in range(deg_list[i]):
                    ports.append(i)
            
            random.shuffle(ports)
            
            edges = set()
            valid = True
            for i in range(0, len(ports), 2):
                u = ports[i]
                v = ports[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
            
            if not valid:
                continue
            
            # Check for C4
            # Build adjacency list
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Check C4
            c4_found = False
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        common = adj[u].intersection(adj[v])
                        if len(common) >= 2:
                            c4_found = True
                            # We could try to repair, but for now just reject
                            break
                if c4_found:
                    break
            
            if not c4_found:
                # Verify min degree
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= n - 39:
                    best_n = n
                    best_edges = [list(e) for e in edges]
                    found = True
                    break
        
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
