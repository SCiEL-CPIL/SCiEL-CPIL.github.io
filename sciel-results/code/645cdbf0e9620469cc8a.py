import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        max_restarts = 500
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use bitmasks for adjacency
            adj = [0] * n  # adj[u] is bitmask of neighbors
            degrees = [0] * n
            edges = []
            
            # can_add(u,v): no C4 created.
            # C4 exists iff exists w in N(u) such that (adj[w] & adj[v]) != 0
            def can_add(u, v):
                if (adj[u] >> v) & 1:
                    return False
                au = adj[u]
                av = adj[v]
                w = au
                while w:
                    lb = w & (-w)
                    w ^= lb
                    wi = lb.bit_length() - 1
                    if adj[wi] & av:
                        return False
                return True
            
            # Greedy construction: repeatedly add edges to the lowest-degree vertex
            progress = True
            while progress and time.time() < deadline:
                progress = False
                min_deg = min(degrees)
                min_verts = [i for i in range(n) if degrees[i] == min_deg]
                random.shuffle(min_verts)
                
                # Try to add multiple edges in this pass
                for u in min_verts:
                    if time.time() > deadline:
                        break
                    au = adj[u]
                    valid_vs = []
                    for v in range(n):
                        if v != u and not ((au >> v) & 1):
                            if can_add(u, v):
                                valid_vs.append(v)
                    
                    if not valid_vs:
                        continue
                    
                    random.shuffle(valid_vs)
                    v = valid_vs[0]
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    edges.append([u, v])
                    progress = True
                    break
            
            if min(degrees) >= target_min_deg:
                return {'n': n, 'edges': edges}
        
        if time.time() > deadline:
            break

    return {'n': 39, 'edges': []}
