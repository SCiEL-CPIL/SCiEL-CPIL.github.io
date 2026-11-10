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

    # Local Search Function
    def local_search(pts, pts_set, edges, max_iters=5000, time_limit=10.0):
        cur_pts = list(pts)
        cur_set = set(pts)
        cur_edges = edges
        pool = [p for p in all_candidates if p not in cur_set]
        
        # Deterministic Best-Improvement Polish
        improved = True
        while improved and (time.time() - t0) < deadline:
            improved = False
            best_delta = 0
            best_move = None
            for i in range(32):
                old = cur_pts[i]
                # Try replacing old with any point in pool
                # Optimization: Only consider points in pool that are "close" or random sample?
                # With 32 points and ~100 candidates, 32*100 = 3200 checks per pass.
                # This is fast.
                for new in pool:
                    delta = edge_delta_remove_add(cur_set - {old}, old, new)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (i, old, new)
            
            if best_move:
                i, old, new = best_move
                cur_pts[i] = new
                cur_set.discard(old)
                cur_set.add(new)
                cur_edges += best_delta
                pool.remove(old)
                pool.append(new)
                improved = True
        
        # Random Search with Perturbation (ILS)
        random.seed(0) # Fixed seed for reproducibility within this function? 
                      # Actually, we want variety. The caller will manage seeds.
        
        stall = 0
        while (time.time() - t0) < deadline and stall < 2000:
            # Random Swap
            i = random.randint(0, 31)
            old = cur_pts[i]
            j = random.randint(0, len(pool) - 1)
            new = pool[j]
            
            if new in cur_set:
                stall += 1
                continue
                
            temp_set = cur_set - {old}
            delta = edge_delta_remove_add(temp_set, old, new)
            
            if delta > 0:
                cur_pts[i] = new
                cur_set.discard(old)
                cur_set.add(new)
                cur_edges += delta
                pool[j] = old
                stall = 0
            else:
                stall += 1
                if stall > 500:
                    # Perturb: Swap 2-3 random points
                    perturb_size = random.randint(2, 3)
                    for _ in range(perturb_size):
                        i1 = random.randint(0, 31)
                        i2 = random.randint(0, 31)
                        if i1 == i2: continue
                        p1 = cur_pts[i1]
                        p2 = cur_pts[i2]
                        # Find two random pool points to swap with
                        j1 = random.randint(0, len(pool) - 1)
                        j2 = random.randint(0, len(pool) - 1)
                        if j1 == j2: continue
                        n1 = pool[j1]
                        n2 = pool[j2]
                        
                        # Check if valid (distinct, not in set)
                        if n1 in cur_set or n2 in cur_set or n1 == n2:
                            continue
                        
                        # Calculate delta for swapping (p1, p2) with (n1, n2)
                        # Remove p1, p2; Add n1, n2
                        temp_set = cur_set - {p1, p2}
                        
                        # Edges lost by removing p1, p2
                        lost = 0
                        for da, db in neighbors:
                            if (p1[0]+da, p1[1]+db) in temp_set: lost += 1
                            if (p2[0]+da, p2[1]+db) in temp_set: lost += 1
                        # Edges gained by adding n1, n2
                        gained = 0
                        for da, db in neighbors:
                            nb1 = (n1[0]+da, n1[1]+db)
                            nb2 = (n2[0]+da, n2[1]+db)
                            if nb1 in temp_set and nb1 != n2: gained += 1
                            if nb2 in temp_set and nb2 != n1: gained += 1
                            if nb1 == n2: gained += 1 # Edge between n1 and n2
                            # Wait, edge between n1 and n2 is counted if they are neighbors.
                            # The logic above counts edges from n1 to existing (temp_set).
                            # If n2 is in temp_set? No, n2 is being added.
                            # So we need to count edges from n1 to n2 separately.
                            # Let's re-evaluate delta calculation for 2-swap.
                        
                        # Simpler: Recalculate edges for the new set?
                        # 32 points, 6 neighbors each. 32*6 = 192 checks.
                        # Very fast.
                        new_pts_list = list(cur_pts)
                        new_pts_list[i1] = n1
                        new_pts_list[i2] = n2
                        new_set = set(new_pts_list)
                        new_edges = count_edges(new_pts_list)
                        
                        delta_2 = new_edges - cur_edges
                        
                        if delta_2 > 0 or random.random() < 0.1: # Accept worse with prob 0.1?
                             # Actually, for ILS, we usually accept the perturbation unconditionally 
                             # to escape, then run LS.
                             pass 
                    # Just do a simple random perturbation: replace 2 random points with 2 random pool points
                    # regardless of delta, to escape basin.
                    i1 = random.randint(0, 31)
                    i2 = random.randint(0, 31)
                    if i1 != i2:
                        j1 = random.randint(0, len(pool) - 1)
                        j2 = random.randint(0, len(pool) - 1)
                        if j1 != j2:
                            n1 = pool[j1]
                            n2 = pool[j2]
                            if n1 not in cur_set and n2 not in cur_set and n1 != n2:
                                cur_pts[i1] = n1
                                cur_pts[i2] = n2
                                cur_set.discard(cur_pts[i1]) # Wrong, we need to remove old ones
                                # Let's do this properly
                                old1 = cur_pts[i1] # This is n1 now? No, we just assigned.
                                # This block is messy. Let's simplify.
                    
                    # Simplified Perturbation:
                    # Pick 2 random indices i, j in cur_pts
                    # Pick 2 random points n1, n2 from pool
                    # Swap them.
                    # Update cur_set, cur_pts, pool.
                    # Recalculate cur_edges.
                    # Reset stall.
                    
                    # Implementation of Perturbation:
                    idx1 = random.randint(0, 31)
                    idx2 = random.randint(0, 31)
                    if idx1 != idx2:
                        pool_idx1 = random.randint(0, len(pool) - 1)
                        pool_idx2 = random.randint(0, len(pool) - 1)
                        if pool_idx1 != pool_idx2:
                            new1 = pool[pool_idx1]
                            new2 = pool[pool_idx2]
                            old1 = cur_pts[idx1]
                            old2 = cur_pts[idx2]
                            
                            # Ensure new1, new2 are not in cur_set
                            if new1 not in cur_set and new2 not in cur_set and new1 != new2:
                                # Apply
                                cur_pts[idx1] = new1
                                cur_pts[idx2] = new2
                                cur_set.discard(old1)
                                cur_set.discard(old2)
                                cur_set.add(new1)
                                cur_set.add(new2)
                                
                                # Update pool
                                pool[pool_idx1] = old1
                                pool[pool_idx2] = old2
                                
                                # Recalculate edges
                                cur_edges = count_edges(cur_pts)
                                stall = 0
                                # Run a quick polish? Or just continue random search.
                                # Continue random search.

        return cur_pts, cur_set, cur_edges

    # Main Loop
    # Start with greedy result
    cur_pts, cur_set, cur_edges = local_search(best_points, best_edges, 1000)
    if cur_edges > best_edges:
        best_edges = cur_edges
        best_points = cur_pts

    # Try multiple ILS iterations
    for k in range(50):
        if (time.time() - t0) > deadline:
            break
        if best_edges >= 76:
            break
            
        # Perturb best solution
        perturbed_pts = list(best_points)
        perturbed_set = set(best_points)
        
        # Randomly swap 2-4 points
        num_swaps = random.randint(2, 4)
        for _ in range(num_swaps):
            i = random.randint(0, 31)
            # Find a random point in pool (points not in perturbed_set)
            # We need a pool for the perturbed set
            # This is getting complicated to maintain a pool dynamically.
            # Alternative: Just generate random candidates and check if they are in set.
            # Given small N, this is fine.
            
            # Let's stick to the previous local_search function which manages its own pool.
            # We will call local_search with a perturbed start.
            
            # How to perturb best_points?
            # Pick 2 random points from best_points, replace with 2 random points from all_candidates not in best_points.
            pool_for_best = [p for p in all_candidates if p not in best_points]
            if len(pool_for_best) < 2: continue
            
            i1 = random.randint(0, 31)
            i2 = random.randint(0, 31)
            if i1 == i2: continue
            
            j1 = random.randint(0, len(pool_for_best) - 1)
            j2 = random.randint(0, len(pool_for_best) - 1)
            if j1 == j2: continue
            
            n1 = pool_for_best[j1]
            n2 = pool_for_best[j2]
            
            perturbed_pts[i1] = n1
            perturbed_pts[i2] = n2
            
            # Check distinctness (n1 != n2, and not equal to other points)
            # n1, n2 are from pool, so not in best_points.
            # But n1 could be equal to n2? No, indices distinct.
            # Could n1 be equal to perturbed_pts[i2] (old)? No, n1 not in best_points.
            # So valid.
            
            perturbed_set = set(perturbed_pts)
            
            # Run local search from this perturbed state
            res_pts, res_set, res_edges = local_search(perturbed_pts, perturbed_set, count_edges(perturbed_pts), 1000)
            
            if res_edges > best_edges:
                best_edges = res_edges
                best_points = res_pts

    return [list(p) for p in best_points]
