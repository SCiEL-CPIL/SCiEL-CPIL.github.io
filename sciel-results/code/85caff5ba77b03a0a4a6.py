import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    def can_add(masks, u, v):
        if (masks[u] >> v) & 1:
            return False
        
        common_uv = masks[u] & masks[v]
        if common_uv.bit_count() > 2:
            return False
            
        # Check neighbors of v (excluding u)
        mask_v = masks[v]
        # We need to check all w in N(v) such that w != u
        # Iterate through set bits
        w = 0
        m = mask_v
        while m:
            if m & 1:
                if w != u:
                    common_uw = masks[u] & masks[w]
                    if common_uw.bit_count() > 1:
                        return False
            m >>= 1
            w += 1
            
        # Check neighbors of u (excluding v)
        mask_u = masks[u]
        w = 0
        m = mask_u
        while m:
            if m & 1:
                if w != v:
                    common_vw = masks[v] & masks[w]
                    if common_vw.bit_count() > 1:
                        return False
            m >>= 1
            w += 1
            
        return True

    while time.time() < deadline:
        random.shuffle(all_edges)
        masks = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            if can_add(masks, u, v):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges

    return best_edges
