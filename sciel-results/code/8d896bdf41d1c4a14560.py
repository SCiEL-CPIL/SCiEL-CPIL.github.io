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
            # C4 u-w-x-v exists iff there's w in N(u), x in N(v), w!=x, and edge(w,x)
            # Equivalent: for some w in N(u), (adj[w] & adj[v]) != 0 and w not in adj[v]
            # Wait: we need w in N(u), x in N(v), w != x, edge(w,x)
            # For each w in N(u): if adj[w] & adj[v] has any bit set that is not w and not u...
            # Actually x must be in N(v) and edge(w,x). x can equal u? No, u not in N(v) yet.
            # x can equal w? No, w!=x required.
            # So: for each w in N(u), check if (adj[w] & adj[v]) has a bit set where the bit is not w.
            # But adj[w] doesn't include w (no self-loops), so adj[w] & adj[v] gives common neighbors of w and v.
            # If w is in N(v), then w is a common neighbor, but that's a triangle not C4.
            # We need x in N(v), x != w, edge(w,x). So (adj[w] & adj[v]) & ~(1<<w) should be nonzero.
            # But adj[w] never has bit w set, so adj[w] & adj[v] never has bit w set.
            # So we just need: exists w in N(u) such that (adj[w] & adj[v]) != 0.
            
            def can_add(u, v):
                if (adj[u] >> v) & 1:
                    return False
                # Check for C4: exists w in N(u) with common neighbor with v (excluding u,v themselves)
                # For each w in N(u): if adj[w] & adj[v] != 0, then C4 exists
                # But we need to be careful: the common neighbor x should not be u or v
                # adj[w] & adj[v]: x is common neighbor of w and v. x != w (no self-loop), x != v.
                # Could x == u? Only if u in adj[w] and u in adj[v]. u in adj[w] means w in N(u) which is true.
                # u in adj[v] means v in N(u)... but we're checking if we CAN add (u,v), so currently v not in N(u).
                # So u is not in adj[v]. Thus x != u automatically.
                # So: C4 exists iff exists w in N(u) such that (adj[w] & adj[v]) != 0
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
            
            # Greedy construction
            progress = True
            while progress and time.time() < deadline:
                progress = False
                min_deg = min(degrees)
                min_verts = [i for i in range(n) if degrees[i] == min_deg]
                random.shuffle(min_verts)
                
                for u in min_verts:
                    # Find valid v
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
