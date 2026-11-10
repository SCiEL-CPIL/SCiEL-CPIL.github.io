import time
import random

def construct():
    t0 = time.time()
    deadline = 140.0
    
    # Neighbors in triangular lattice (axial coordinates)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def count_edges(pts):
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a + da, b + db) in pset:
                    edges += 1
        return edges // 2

    def edge_delta_remove_add(pts_set, remove_pt, add_pt):
        lost = 0
        for da, db in neighbors:
            if (remove_pt[0] + da, remove_pt[1] + db) in pts_set:
                lost += 1
        gained = 0
        for da, db in neighbors:
            nb = (add_pt[0] + da, add_pt[1] + db)
            if nb in pts_set and nb != remove_pt:
                gained += 1
        return gained - lost

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    # --- Phase 1: Deterministic Greedy Spiral Construction ---
    # Candidate pool: all points within hex_dist <= 6
    all_candidates = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if hex_dist(a, b) <= 6:
                all_candidates.append((a, b))
    
    # Greedy spiral construction
    current = [(0, 0)]
    current_set = {(0, 0)}
    
    for _ in range(31):
        best_pt = None
        best_score = -1
        best_dist = 999
        for pt in all_candidates:
            if pt in current_set:
                continue
            # Count edges this point would add
            score = 0
            for da, db in neighbors:
                if (pt[0] + da, pt[1] + db) in current_set:
                    score += 1
            d = hex_dist(pt[0], pt[1])
            if score > best_score or (score == best_score and d < best_dist):
                best_score = score
                best_dist = d
                best_pt = pt
        if best_pt:
            current.append(best_pt)
            current_set.add(best_pt)
    
    best_points = list(current)
    best_edges = count_edges(best_points)
    
    if best_edges >= 76:
        return [list(p) for p in best_points]
    
    # --- Phase 2: Local Search (Hill Climbing) ---
    # Build a pool of potential replacement points
    # We want points that are likely to be useful, so we stick to a reasonable range
    # hex_dist <= 6 is sufficient for n=32
    pool = [p for p in all_candidates if p not in current_set]
    
    random.seed(42)
    current = list(best_points)
    current_set = set(current)
    cur_edges = best_edges
    
    stall = 0
    stall_limit = 500
    
    while (time.time() - t0) < deadline:
        if stall > stall_limit:
            break
        
        i = random.randint(0, 31)
        old = current[i]
        j = random.randint(0, len(pool) - 1)
        new = pool[j]
        
        if new in current_set:
            stall += 1
            continue
            
        temp_set = current_set - {old}
        delta = edge_delta_remove_add(temp_set, old, new)
        
        if delta > 0:
            current[i] = new
            current_set.discard(old)
            current_set.add(new)
            cur_edges += delta
            stall = 0
            pool[j] = old
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_points = list(current)
                if best_edges >= 76:
                    break
        else:
            stall += 1
    
    if best_edges >= 76:
        return [list(p) for p in best_points]
    
    # --- Phase 3: More Restarted Local Search if needed ---
    # If we are stuck, we can try different random seeds or slightly different initial states
    # But since the greedy start is very good, simple restarts from the best found so far
    # with different random seeds for the swaps should work.
    
    for seed in range(20):
        if (time.time() - t0) > deadline:
            break
            
        random.seed(seed)
        current = list(best_points)
        current_set = set(current)
        cur_edges = best_edges
        stall = 0
        
        # Rebuild pool based on current best_points
        pool = [p for p in all_candidates if p not in current_set]
        
        while (time.time() - t0) < deadline and stall < 1000:
            i = random.randint(0, 31)
            old = current[i]
            j = random.randint(0, len(pool) - 1)
            new = pool[j]
            
            if new in current_set:
                stall += 1
                continue
                
            temp_set = current_set - {old}
            delta = edge_delta_remove_add(temp_set, old, new)
            
            if delta > 0:
                current[i] = new
                current_set.discard(old)
                current_set.add(new)
                cur_edges += delta
                stall = 0
                pool[j] = old
                
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_points = list(current)
                    if best_edges >= 76:
                        break
            else:
                stall += 1
                
        if best_edges >= 76:
            break

    return [list(p) for p in best_points]
