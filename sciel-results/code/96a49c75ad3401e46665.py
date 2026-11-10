import time

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
    
    # Generate multiple spiral initializations by rotating the greedy choice
    # We will try 6 rotations of the neighbor order to break symmetries
    best_points = []
    best_edges = 0
    
    for rot in range(6):
        # Rotate neighbor list to change greedy preference
        rotated_neighbors = neighbors[rot:] + neighbors[:rot]
        
        current = [(0, 0)]
        current_set = {(0, 0)}
        
        for _ in range(31):
            best_pt = None
            best_score = -1
            best_dist = 999
            for pt in all_candidates:
                if pt in current_set:
                    continue
                # Count edges this point would add using rotated neighbors
                score = 0
                for da, db in rotated_neighbors:
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
        
        cur_edges = count_edges(current)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_points = list(current)
            
        if best_edges >= 76:
            return [list(p) for p in best_points]

    # --- Phase 2: Deterministic Best-Improvement Hill Climbing ---
    # This is much more reliable than random swaps for small N.
    # We iterate through all possible (remove, add) pairs and pick the best gain.
    # If no gain is possible, we stop (or could implement simulated annealing, but usually not needed for this size).
    
    current = list(best_points)
    current_set = set(current)
    cur_edges = best_edges
    
    # Pool of candidates to add (excluding current points)
    # We use a slightly larger pool to allow for "jumps" if needed, but hex_dist <= 6 is usually enough.
    # Let's stick to hex_dist <= 7 to be safe but still fast.
    pool = [p for p in all_candidates if p not in current_set]
    
    # To allow for more flexibility, let's expand the pool slightly if we don't reach 76
    # But first, try with the tight pool.
    
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        best_delta = 0
        best_remove_idx = -1
        best_add_pt = None
        
        # Iterate over all points to remove
        for i in range(32):
            old = current[i]
            # Temporarily remove old from set for delta calculation
            # We can calculate delta without modifying the set for efficiency in the inner loop
            # but modifying is O(1) amortized. Let's just use the helper which takes the set.
            # We need the set without 'old' for the 'gained' part?
            # Actually, edge_delta_remove_add(pts_set, remove_pt, add_pt) assumes pts_set contains remove_pt?
            # Let's check the helper:
            # lost: neighbors of remove_pt in pts_set. (Correct, remove_pt is in pts_set)
            # gained: neighbors of add_pt in pts_set, excluding remove_pt.
            # So we can pass the current set directly.
            
            temp_set = current_set
            # Optimization: Pre-calculate lost edges for 'old'
            lost = 0
            for da, db in neighbors:
                if (old[0] + da, old[1] + db) in temp_set:
                    lost += 1
            
            # Iterate over all candidates to add
            for new in pool:
                if new in current_set:
                    continue
                
                gained = 0
                for da, db in neighbors:
                    nb = (new[0] + da, new[1] + db)
                    if nb in temp_set and nb != old:
                        gained += 1
                
                delta = gained - lost
                if delta > best_delta:
                    best_delta = delta
                    best_remove_idx = i
                    best_add_pt = new
        
        if best_delta > 0:
            # Apply the best move
            old = current[best_remove_idx]
            new = best_add_pt
            
            current_set.discard(old)
            current_set.add(new)
            current[best_remove_idx] = new
            cur_edges += best_delta
            
            # Update pool: remove 'new' from pool, add 'old' to pool
            # This maintains the invariant that pool = candidates \ current_set
            # O(N) update is fine for small N.
            pool.remove(new)
            pool.append(old)
            
            improved = True
            
            if cur_edges >= 76:
                break
    
    # If we still haven't reached 76, we might be stuck in a local minimum.
    # A simple "kick" or restart from a slightly perturbed state could help.
    # However, the spiral + best-improvement usually finds 76 for n=32.
    
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_points = list(current)

    return [list(p) for p in best_points]
