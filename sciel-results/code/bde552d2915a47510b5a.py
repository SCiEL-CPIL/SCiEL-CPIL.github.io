import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for 150s limit
    
    best_n = 0
    best_edges = []
    
    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        if time.time() > deadline:
            break
            
        d_req = n - 39
        
        # Try multiple random orderings of edge additions
        max_restarts = 300 if n <= 45 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Generate all possible edges
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            # adj[v] is a 64-bit mask of neighbors of v
            adj = [0] * n
            deg = [0] * n
            
            # Greedy edge addition in random order
            for u, v in all_edges:
                # Check C4: count common neighbors of u and v
                common = adj[u] & adj[v]
                # If common has 0 or 1 bits set, it's safe to add
                # If 2 or more, adding (u,v) would create a C4
                if common == 0 or (common & (common - 1)) == 0:
                    # Safe to add (0 or 1 common neighbor)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
            
            # Check if minimum degree requirement is met
            min_deg = min(deg) if deg else 0
            if min_deg >= d_req:
                # Verify no C4s
                valid = True
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u] & adj[v]
                        # If common has 2 or more bits, there's a C4
                        if common != 0 and (common & (common - 1)) != 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Extract edges
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    
                    return {'n': n, 'edges': edges}
        
        # If we didn't find for this n, continue to lower n
    
    # Fallback: return best found or trivial
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Trivial valid graph
    return {'n': 1, 'edges': []}
