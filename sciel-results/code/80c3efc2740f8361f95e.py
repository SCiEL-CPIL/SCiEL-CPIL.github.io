import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 46
    min_deg = n - 39  # 7
    
    # We will try to construct a graph on 46 vertices.
    # If it fails, we fall back to 39.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        deg = [0] * n
        
        # Symmetry breaking: Fix vertex 0's neighbors to 1..7
        adj[0] = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        for v in range(1, 8):
            adj[v] |= (1 << 0)
            deg[v] += 1
        deg[0] = 7
        
        success = True
        while True:
            if time.time() > deadline:
                success = False
                break
                
            deficient = [i for i in range(n) if deg[i] < min_deg]
            if not deficient:
                break
                
            # Prioritize vertices with lowest degree
            u = min(deficient, key=lambda x: deg[x])
            
            valid_neighbors = []
            for v in range(n):
                if v == u:
                    continue
                if (adj[u] >> v) & 1:
                    continue
                if (adj[u] & adj[v]) == 0:
                    valid_neighbors.append(v)
                    
            if not valid_neighbors:
                success = False
                break
                
            # Prefer neighbors that are also deficient
            deficient_set = set(deficient)
            deficient_valid = [v for v in valid_neighbors if v in deficient_set]
            if deficient_valid:
                v = random.choice(deficient_valid)
            else:
                v = random.choice(valid_neighbors)
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        if success:
            # Final verification
            valid = True
            for i in range(n):
                if deg[i] < min_deg:
                    valid = False
                    break
            if valid:
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            if (adj[u] & adj[v]) != 0:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                    
    return {'n': 39, 'edges': []}
