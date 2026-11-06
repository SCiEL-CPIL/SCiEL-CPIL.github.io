import numpy as np
import random
import time
from collections import Counter

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    """
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin
    
    def compute_all_areas(pts):
        """
        Compute all triangle areas for a set of points.
        Returns:
            areas: array of all non-zero doubled areas
            counts: Counter of areas
            best_area: area with max count
            best_count: max count
        """
        n = len(pts)
        if n < 3:
            return np.array([]), Counter(), 0, 0
            
        pts_arr = np.array(pts, dtype=np.int64)
        x = pts_arr[:, 0]
        y = pts_arr[:, 1]
        
        # Use combinations to avoid full 3D meshgrid if possible, 
        # but for N=30, meshgrid is fine and fast enough in NumPy.
        # However, generating indices for i<j<k is cleaner.
        
        indices = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    indices.append((i, j, k))
                    
        if not indices:
            return np.array([]), Counter(), 0, 0
            
        idx_arr = np.array(indices)
        ii = idx_arr[:, 0]
        jj = idx_arr[:, 1]
        kk = idx_arr[:, 2]
        
        xi = x[ii]
        yi = y[ii]
        xj = x[jj]
        yj = y[jj]
        xk = x[kk]
        yk = y[kk]
        
        # Doubled area = |det|
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Exclude zero areas
        mask = areas > 0
        areas = areas[mask]
        
        if len(areas) == 0:
            return np.array([]), Counter(), 0, 0
            
        counts = Counter(areas.tolist())
        best_area, best_count = counts.most_common(1)[0]
        
        return areas, counts, int(best_area), int(best_count)

    def compute_areas_for_point(pts, idx):
        """
        Compute areas of all triangles involving pts[idx].
        Returns array of areas (including zeros).
        """
        n = len(pts)
        pts_arr = np.array(pts, dtype=np.int64)
        pi = pts_arr[idx]
        others = np.delete(pts_arr, idx, axis=0)
        
        if len(others) < 2:
            return np.array([])
            
        # Get all pairs from others
        j_idx, k_idx = np.meshgrid(np.arange(len(others)), np.arange(len(others)), indexing='ij')
        mask = j_idx < k_idx
        jj = j_idx[mask]
        kk = k_idx[mask]
        
        x0, y0 = pi
        xj = others[jj, 0]
        yj = others[jj, 1]
        xk = others[kk, 0]
        yk = others[kk, 1]
        
        areas = np.abs(x0 * (yj - yk) + xj * (yk - y0) + xk * (y0 - yj))
        return areas

    def compute_areas_for_new_point(new_point, others_arr):
        """
        Compute areas of all triangles involving new_point with the given others.
        others_arr should be a numpy array of shape (n-1, 2).
        """
        if len(others_arr) < 2:
            return np.array([])
            
        j_idx, k_idx = np.meshgrid(np.arange(len(others_arr)), np.arange(len(others_arr)), indexing='ij')
        mask = j_idx < k_idx
        jj = j_idx[mask]
        kk = k_idx[mask]
        
        x0, y0 = new_point
        xj = others_arr[jj, 0]
        yj = others_arr[jj, 1]
        xk = others_arr[kk, 0]
        yk = others_arr[kk, 1]
        
        areas = np.abs(x0 * (yj - yk) + xj * (yk - y0) + xk * (y0 - yj))
        return areas

    # --- Initialization ---
    
    candidates = []
    
    # Pattern 1: Two parallel rows
    # Maximize length to get many small areas
    for y0 in range(-15, 16):
        for gap in [1, 2, 3, 4, 5, 6]:
            y1 = y0 + gap
            if -30 <= y1 <= 30:
                # Try to fit 15 points in each row
                for x_start in range(-20, 16):
                    pts = []
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)
                        
    # Pattern 2: Three parallel rows
    for y0 in range(-15, 16):
        for gap in [1, 2, 3]:
            y1 = y0 + gap
            y2 = y0 + 2 * gap
            if -30 <= y2 <= 30:
                for x_start in range(-20, 16):
                    pts = []
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y1))
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y2))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)

    # Pattern 3: Grids
    for w in range(5, 9):
        for h in range(5, 9):
            if w * h >= 30:
                for ox in range(-20, 16):
                    for oy in range(-20, 16):
                        pts = []
                        for x in range(w):
                            for y in range(h):
                                px = ox + x
                                py = oy + y
                                if -30 <= px <= 30 and -30 <= py <= 30:
                                    pts.append((px, py))
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if len(set(pts)) == 30:
                                candidates.append(pts)

    # Pattern 4: Diagonal lines (for more diverse area distributions)
    for slope in [1, -1, 2, -2]:
        for intercept in range(-20, 21):
            pts = []
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) >= 30:
                # Take a subset
                random.shuffle(pts)
                pts = pts[:30]
                if len(set(pts)) == 30:
                    candidates.append(pts)

    if not candidates:
        # Fallback
        candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    random.seed(42)
    best_score = 0
    best_points = None
    best_counts = None
    best_area = 0
    
    # Evaluate initial candidates (limit to avoid timeout)
    max_candidates_to_eval = 500
    eval_count = 0
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 10:  # Leave 10s for local search
            break
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        _, counts, area, count = compute_all_areas(cand)
        if count > best_score:
            best_score = count
            best_points = cand[:]
            best_counts = counts
            best_area = area
        eval_count += 1
        if eval_count >= max_candidates_to_eval:
            break

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        _, best_counts, best_area, best_score = compute_all_areas(best_points)
    
    # Additional: Try a few more structured patterns if we have time
    if time.time() - start_time < DEADLINE - 15:
        extra_patterns = []
        
        # Concentric circles (approximated with integer points)
        for r in [3, 4, 5]:
            pts = []
            for x in range(-r, r+1):
                for y in range(-r, r+1):
                    if x*x + y*y == r*r:
                        pts.append((x, y))
            if len(pts) >= 30:
                random.shuffle(pts)
                pts = pts[:30]
                if len(set(pts)) == 30:
                    extra_patterns.append(pts)
        
        # Regular polygons
        for n_sides in [6, 8, 10, 12]:
            for r in [5, 8, 10]:
                pts = []
                for i in range(n_sides * 3):
                    angle = 2 * np.pi * i / (n_sides * 3)
                    x = int(round(r * np.cos(angle)))
                    y = int(round(r * np.sin(angle)))
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
                if len(pts) >= 30:
                    # Deduplicate
                    seen = set()
                    unique_pts = []
                    for p in pts:
                        if p not in seen:
                            seen.add(p)
                            unique_pts.append(p)
                    if len(unique_pts) >= 30:
                        random.shuffle(unique_pts)
                        unique_pts = unique_pts[:30]
                        extra_patterns.append(unique_pts)
        
        for cand in extra_patterns:
            if time.time() - start_time > DEADLINE - 10:
                break
            if len(cand) != 30 or len(set(cand)) != 30:
                continue
            _, counts, area, count = compute_all_areas(cand)
            if count > best_score:
                best_score = count
                best_points = cand[:]
                best_counts = counts
                best_area = area

    # --- Local Search ---
    
    # Precompute areas for each point for fast delta calculation
    # areas_per_point[i] is an array of areas for triangles involving point i
    areas_per_point = [compute_areas_for_point(best_points, i) for i in range(30)]
    
    current_counts = best_counts.copy()
    current_best_area = best_area
    current_best_count = best_score
    current_points = best_points[:]
    
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Helper to update counts
    def update_counts(counts, areas_to_remove, areas_to_add):
        for a in areas_to_remove:
            if a > 0:
                counts[a] -= 1
                if counts[a] == 0:
                    del counts[a]
        for a in areas_to_add:
            if a > 0:
                counts[a] += 1
                
    # Local search loop
    no_improvement_counter = 0
    max_no_improvement = 200
    
    # Precompute the "others" arrays for each point index
    # others_for_i[i] = array of all points except current_points[i]
    # We'll recompute this when a point changes
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Shuffle order of points to try
        point_order = list(range(30))
        random.shuffle(point_order)
        
        for i in point_order:
            if time.time() - start_time > DEADLINE:
                break
                
            old_point = current_points[i]
            old_areas = areas_per_point[i]
            
            # Count how many of old_areas match current_best_area
            removed_best = np.sum(old_areas == current_best_area)
            
            # Precompute others array once for this i
            others_arr = np.array([p for j, p in enumerate(current_points) if j != i], dtype=np.int64)
            
            # Generate candidates
            candidates_to_try = []
            
            # 1. Local perturbations (wider range)
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx = old_point[0] + dx
                    ny = old_point[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        np_ = (nx, ny)
                        if np_ != old_point and np_ not in current_points:
                            candidates_to_try.append(np_)
            
            # 2. Random valid points
            for _ in range(15):
                np_ = random.choice(all_valid_points)
                if np_ != old_point and np_ not in current_points:
                    candidates_to_try.append(np_)
            
            random.shuffle(candidates_to_try)
            
            for new_point in candidates_to_try[:30]:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Compute new areas using the precomputed others_arr
                new_areas = compute_areas_for_new_point(new_point, others_arr)
                
                if len(new_areas) == 0:
                    continue
                
                # Check if this move improves the best count
                added_best = np.sum(new_areas == current_best_area)
                new_count = current_best_count - removed_best + added_best
                
                if new_count > current_best_count:
                    # Accept move
                    # Update counts
                    update_counts(current_counts, old_areas, new_areas)
                    
                    # Update state
                    current_points[i] = new_point
                    areas_per_point[i] = new_areas
                    current_best_count = new_count
                    
                    improved = True
                    no_improvement_counter = 0
                    break # Move to next point
                
            if improved:
                # Check if best_area changed (just in case another area became dominant)
                if current_counts:
                    new_best_area, new_best_count = current_counts.most_common(1)[0]
                    if new_best_count > current_best_count:
                        current_best_area = new_best_area
                        current_best_count = new_best_count
                    elif new_best_count == current_best_count and new_best_area != current_best_area:
                        # Tie: switch to the new area to potentially escape local optima
                        current_best_area = new_best_area
                        current_best_count = new_best_count
                continue # Continue to next point
        
        if not improved:
            no_improvement_counter += 1
            if no_improvement_counter > max_no_improvement:
                break

    return current_points
