import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    def _score(points):
        """Return (max_multiplicity, best_area) for a point set."""
        n = len(points)
        if n < 3:
            return 0, 0
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = min(a for a, m in areas.items() if m == max_mult)
        return max_mult, max_area

    def _calc_delta(points, idx, new_pos, target_area):
        """
        Calculate the change in multiplicity for `target_area` when moving point `idx` to `new_pos`.
        Returns delta_count.
        """
        old_x, old_y = points[idx]
        new_x, new_y = new_pos
        delta = 0
        
        # Iterate over all other pairs (j, k) where j != idx and k != idx
        # Since we move one point, we only need to check triangles involving that point.
        # A triangle (i, j, k) is affected if i == idx.
        # We need to check all pairs (j, k) from the remaining 29 points.
        # There are C(29, 2) = 406 such pairs.
        
        # To avoid O(N^2) overhead in Python loops for every single step, 
        # we rely on the fact that N=30 is small, but 406 checks * many steps is still fast.
        
        for j in range(30):
            if j == idx:
                continue
            xj, yj = points[j]
            for k in range(j + 1, 30):
                if k == idx:
                    continue
                xk, yk = points[k]
                
                # Area with old point
                det_old = abs((xj - old_x) * (yk - old_y) - (yj - old_y) * (xk - old_x))
                # Area with new point
                det_new = abs((xj - new_x) * (yk - new_y) - (yj - new_y) * (xk - new_x))
                
                if det_old == target_area:
                    delta -= 1
                if det_new == target_area:
                    delta += 1
        return delta

    def _local_search(points, target_area, max_time):
        """
        Optimize the count of triangles with area == target_area.
        """
        current_points = list(points)
        
        # Calculate initial count
        current_count = 0
        n = 30
        for i in range(n):
            x1, y1 = current_points[i]
            for j in range(i + 1, n):
                x2, y2 = current_points[j]
                for k in range(j + 1, n):
                    x3, y3 = current_points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det == target_area:
                        current_count += 1
        
        best_count = current_count
        best_points = list(current_points)
        
        # Precompute a set of occupied points for fast collision check
        occupied = set(current_points)
        
        T = 5.0
        alpha = 0.999
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < max_time and no_improve < max_no_improve:
            idx = random.randint(0, n - 1)
            old_x, old_y = current_points[idx]
            
            # Try a few random moves
            found_move = False
            for _ in range(10):
                step = random.choice([1, 1, 1, 2, 2, 3])
                dx = random.randint(-step, step)
                dy = random.randint(-step, step)
                if dx == 0 and dy == 0:
                    continue
                
                new_x = old_x + dx
                new_y = old_y + dy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pos = (new_x, new_y)
                
                # Collision check
                if new_pos in occupied:
                    # Check if it's the same point (shouldn't happen with dx,dy != 0)
                    continue
                
                # Calculate delta
                delta = _calc_delta(current_points, idx, new_pos, target_area)
                
                if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                    # Accept move
                    occupied.remove(old_pos)
                    occupied.add(new_pos)
                    current_points[idx] = new_pos
                    current_count += delta
                    found_move = True
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_points = list(current_points)
                        no_improve = 0
                    else:
                        no_improve += 1
                    break
                else:
                    # Reject move, try next random perturbation
                    pass
            
            if not found_move:
                no_improve += 1
            
            T *= alpha
            if T < 0.01:
                T = 0.01
        
        return best_points, best_count

    # 1. Generate diverse initial candidates
    candidates = []
    
    # A. Rectangular Grids (Standard)
    for w, h in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        pts_c = [(x - cx, y - cy) for x, y in pts]
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
            candidates.append(pts_c)
            
    # B. Sheared Grids
    random.seed(42)
    for s in range(-5, 6):
        m, p = 6, 5
        pts = []
        for i in range(m):
            for j in range(p):
                x = i + s * j
                y = j
                pts.append((x, y))
        if len(set(pts)) == 30:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            pts_c = [(x - cx, y - cy) for x, y in pts]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                candidates.append(pts_c)

    # C. Lattice Points (Arithmetic Progressions)
    # Points: (a*i + b*j, c*i + d*j)
    # Small basis vectors to keep points close
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    det = a * d - b * c
                    if det == 0:
                        continue
                    # Use 6x5 grid in lattice space
                    pts = []
                    for i in range(6):
                        for j in range(5):
                            x = a * i + b * j
                            y = c * i + d * j
                            pts.append((x, y))
                    if len(set(pts)) == 30:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        cx = (min(xs) + max(xs)) // 2
                        cy = (min(ys) + max(ys)) // 2
                        pts_c = [(x - cx, y - cy) for x, y in pts]
                        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                            candidates.append(pts_c)
    
    # D. Random Clusters
    for seed in range(20):
        random.seed(seed)
        cx = random.randint(-5, 5)
        cy = random.randint(-5, 5)
        spread = random.randint(5, 10)
        pts = set()
        while len(pts) < 30:
            x = cx + random.randint(-spread, spread)
            y = cy + random.randint(-spread, spread)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        candidates.append(list(pts))

    # Evaluate candidates to find top ones
    scored_candidates = []
    for pts in candidates:
        score, area = _score(pts)
        scored_candidates.append((score, area, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    if not scored_candidates:
        return [(i % 6, i // 6) for i in range(30)]
    
    best_score = scored_candidates[0][0]
    best_points = scored_candidates[0][2]
    
    # 2. Run Local Search on top candidates
    # We will target the area that currently has the highest multiplicity in the best candidate,
    # and also try small areas (1, 2, 3) which are common in dense grids.
    
    top_candidates = scored_candidates[:10]
    
    # Determine which areas to target
    target_areas = set()
    for sc, ar, pts in top_candidates:
        target_areas.add(ar)
    # Always try small areas
    target_areas.update([1, 2, 3, 4, 5])
    
    # Allocate time for local search
    # We have ~55s total. Scoring took some time. Let's reserve 5s for final check.
    ls_deadline = start_time + 50
    
    # Sort target areas by frequency in top candidates? Or just iterate?
    # Let's iterate over top candidates and for each, try optimizing its best area.
    
    for sc, ar, pts in top_candidates:
        if time.time() > ls_deadline - 5:
            break
            
        # For this candidate, try to improve its specific area `ar`
        # Also try area 1 if ar != 1, as area 1 is often the max for dense grids
        areas_to_try = [ar]
        if ar != 1:
            areas_to_try.append(1)
            
        for target_area in areas_to_try:
            if time.time() > ls_deadline - 2:
                break
                
            # Run local search with a time slice
            # Give each LS run 2-3 seconds
            ls_end = time.time() + 3
            if ls_end > ls_deadline - 1:
                ls_end = ls_deadline - 1
                
            improved_pts, improved_count = _local_search(pts, target_area, ls_end)
            
            # Verify the improvement with full score
            # The local search optimized for `target_area`, but the global max multiplicity
            # might be for a different area, or the count for target_area might not be the max.
            full_score, full_area = _score(improved_pts)
            
            if full_score > best_score:
                best_score = full_score
                best_points = improved_pts
            elif full_score == best_score and full_area < scored_candidates[0][1]:
                # Tie-breaker: smaller area is better for stability? 
                # Actually, the task says "maximize the largest multiplicity". 
                # If multiplicities are equal, the score is the same.
                # But smaller area might be more robust? 
                # Let's stick to strictly greater score for now to avoid noise.
                pass

    # Final check
    return best_points

