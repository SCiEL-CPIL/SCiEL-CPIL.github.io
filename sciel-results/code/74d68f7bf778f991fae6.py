import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 3000
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            # Helper: check if adding (u,v) creates a C4
            # C4 exists iff there's a path u-w-x-v of length 3
            # i.e., ∃w∈N(u) such that adj[w] & adj[v] ≠ 0
            # (before adding the edge)
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False  # already connected
                # For each w in N(u), check if adj[w] intersects N(v)
                mask_u = adj[u]
                mask_v = adj[v]
                m = mask_u
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if adj[w] & mask_v:
                        return False
                    m ^= lsb
                return True
            
            # Greedy construction biased toward low-degree vertices
            # We'll try to add edges until we can't or time runs out
            # At each step, pick a random vertex with lowest degree, then a random partner
            
            # To speed up, we can batch: try many random pairs
            # But bias toward low-degree vertices
            
            # Strategy: maintain a priority on low-degree vertices
            # Simple approach: repeatedly pick random u from low-degree set, random v
            
            # We'll do a loop: try to add edges, prioritizing vertices below target
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                
                # Find vertices below target degree
                low_verts = [i for i in range(n) if degrees[i] < min_deg]
                
                if not low_verts:
                    break
                
                # Randomly pick a low-degree vertex
                u = random.choice(low_verts)
                
                # Try random partners
                # Shuffle potential partners
                partners = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                random.shuffle(partners)
                
                for v in partners:
                    if can_add(u, v):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added_any = True
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    # For n=39, min_deg=0, empty graph works
    return {'n': 39, 'edges': []}
