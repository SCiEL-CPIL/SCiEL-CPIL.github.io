import time
import random

def construct():
    start_time = time.time()
    timeout = 145.0  # 150s limit, keep 5s margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def popcount(x):
        return bin(x).count("1")
        
    def is_valid(u, v, masks):
        # masks is a list of integers
        # 1. Check (u, v)
        if popcount(masks[u] & masks[v]) > 2:
            return False
        
        # 2. Check (u, x) for x in neighbors[u]
        # Iterate through bits set in masks[u]
        nu = masks[u]
        # We need to check all x in N(u).
        # It's faster to just iterate 0..n-1 and check if x is in N(u)
        for x in range(n):
            if x == u or x == v: continue
            if (nu >> x) & 1:
                # x is in N(u)
                common_ux = popcount(masks[u] & masks[x])
                if (masks[x] >> v) & 1:
                    if common_ux > 1:
                        return False
                else:
                    if common_ux > 2:
                        return False
                        
        # 3. Check (v, x) for x in neighbors[v]
        nv = masks[v]
        for x in range(n):
            if x == u or x == v: continue
            if (nv >> x) & 1:
                # x is in N(v)
                common_vx = popcount(masks[v] & masks[x])
                if (masks[x] >> u) & 1:
                    if common_vx > 1:
                        return False
                else:
                    if common_vx > 2:
                        return False
                        
        return True

    # Randomized Greedy
    # Increase restarts if time allows
    for i in range(20000): 
        if time.time() - start_time > timeout:
            break
        
        random.shuffle(all_edges)
        masks = [0] * n
        edges = []
        
        for u, v in all_edges:
            if is_valid(u, v, masks):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
