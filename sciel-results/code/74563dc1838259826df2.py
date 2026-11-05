import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    def try_construct(n, time_limit):
        """Try to construct a C4-free graph on n vertices with min degree >= n-39."""
        min_deg_req = n - 39
        if min_deg_req < 0:
            min_deg_req = 0
        
        # Use bitmask representation for adjacency
        # adj[i] is a bitmask of neighbors of i
        
        best_deg = 0
        best_adj = None
        
        for attempt in range(500):
            if time.time() > time_limit:
                break
            
            random.seed(attempt * 12345 + 7)
            
            adj = [0] * n
            deg = [0] * n
            
            # Greedy: repeatedly add edges that don't create C4
            # Prioritize vertices with lowest degree
            max_iter = 20000
            stuck = 0
            
            for _ in range(max_iter):
                if time.time() > time_limit:
                    break
                
                # Check if all vertices meet min degree
                if all(d >= min_deg_req for d in deg):
                    break
                
                # Find vertices below min degree
                low_verts = [i for i in range(n) if deg[i] < min_deg_req]
                if not low_verts:
                    break
                
                # Pick the vertex with the lowest degree (with random tie-breaking)
                min_d = min(deg[i] for i in low_verts)
                candidates = [i for i in low_verts if deg[i] == min_d]
                u = random.choice(candidates)
                
                # Find valid v to connect to u
                # A connection (u,v) is safe if N(u) and N(v) have no common neighbor
                # i.e., adj[u] & adj[v] == 0
                valid_vs = []
                for v in range(n):
                    if v == u:
                        continue
                    if adj[u] >> v & 1:
                        continue
                    # Check C4 safety: no common neighbor
                    if adj[u] & adj[v] == 0:
                        valid_vs.append(v)
                
                if not valid_vs:
                    stuck += 1
                    if stuck > 100:
                        break
                    continue
                
                stuck = 0
                # Pick v with lowest degree to balance
                v = min(valid_vs, key=lambda x: deg[x])
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
            
            # Check if we succeeded
            if all(d >= min_deg_req for d in deg):
                return adj, True
            
            # Track best attempt
            min_d = min(deg)
            if min_d > best_deg:
                best_deg = min_d
                best_adj = adj[:]
        
        return best_adj, False
    
    # Try n=47 first
    if time.time() < deadline:
        adj, success = try_construct(47, deadline - 5)
        if success:
            edges = []
            for i in range(47):
                for j in range(i+1, 47):
                    if adj[i] >> j & 1:
                        edges.append([i, j])
            return {'n': 47, 'edges': edges}
    
    # Try n=46
    if time.time() < deadline:
        adj, success = try_construct(46, deadline - 5)
        if success:
            edges = []
            for i in range(46):
                for j in range(i+1, 46):
                    if adj[i] >> j & 1:
                        edges.append([i, j])
            return {'n': 46, 'edges': edges}
    
    # Fallback: try smaller n
    for n in [45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 10:
            break
        adj, success = try_construct(n, deadline - 5)
        if success:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] >> j & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Final fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}
