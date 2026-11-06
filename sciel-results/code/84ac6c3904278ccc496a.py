import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute distances
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    dist2_np = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.uint16)
    for i in range(NUM_POINTS):
        dx = xs[i] - xs
        dy = ys[i] - ys
        dist2_np[i] = dx*dx + dy*dy
        
    # Precompute circles
    circles = [dict() for _ in range(NUM_POINTS)]
    for u in range(NUM_POINTS):
        for v in range(u + 1, NUM_POINTS):
            d2 = dist2_np[u][v]
            circles[u].setdefault(d2, 0)
            circles[u][d2] |= (1 << v)
            circles[v].setdefault(d2, 0)
            circles[v][d2] |= (1 << u)
            
    best_set = []
    best_size = 0
    
    # Greedy Initial Solution
    current_set = []
    forbidden = 0
    available = (1 << NUM_POINTS) - 1
    
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    
    for u in order:
        if u & forbidden:
            continue
        current_set.append(u)
        for v in current_set[:-1]:
            d2 = dist2_np[u][v]
            c1 = circles[u].get(d2, 0)
            c2 = circles[v].get(d2, 0)
            mask = (dist2_np[u] == dist2_np[v])
            c3 = int.from_bytes(mask.tobytes(), 'little')
            forbidden |= (c1 | c2 | c3)
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    # B&B
    def dfs(available, forbidden, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        candidates = available & ~forbidden
        if candidates == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        if len(current_set) + candidates.bit_count() <= best_size:
            return
            
        # Get candidate indices
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lsb
            
        # Heuristic: Pick candidate with max impact
        # Limit to first 20 to save time
        search_cands = cand_list[:20]
        best_u = search_cands[0]
        max_impact = -1
        
        for u in search_cands:
            impact = 0
            for v in current_set:
                d2 = dist2_np[u][v]
                c1 = circles[u].get(d2, 0)
                c2 = circles[v].get(d2, 0)
                # Approximate impact without C3 for speed in heuristic
                impact += (c1 | c2).bit_count()
            if impact > max_impact:
                max_impact = impact
                best_u = u
                
        u = best_u
        
        # Branch 1: Include u
        new_forbidden = forbidden
        for v in current_set:
            d2 = dist2_np[u][v]
            c1 = circles[u].get(d2, 0)
            c2 = circles[v].get(d2, 0)
            mask = (dist2_np[u] == dist2_np[v])
            c3 = int.from_bytes(mask.tobytes(), 'little')
            new_forbidden |= (c1 | c2 | c3)
            
        new_available = available & ~(1 << u)
        dfs(new_available, new_forbidden, current_set + [u])
        
        # Branch 2: Exclude u
        dfs(new_available, forbidden, current_set)
        
    dfs(available, forbidden, current_set)
    
    # Convert indices to points
    points = [(x, y) for x in range(N) for y in range(N)]
    return [points[i] for i in best_set]
