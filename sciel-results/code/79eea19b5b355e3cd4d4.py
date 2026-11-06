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
    DEADLINE = 57.0  # Safety margin
    
    def compute_all_areas(pts):
        """
        Compute all triangle areas for a set of points.
        Returns:
            counts: Counter of areas
            best_area: area with max count
            best_count: max count
        """
        n = len(pts)
        if n < 3:
            return Counter(), 0, 0
            
        pts_arr = np.array(pts, dtype=np.int64)
        
        counts = Counter()
        for i in range(n):
            pi = pts_arr[i]
            others = pts_arr[i+1:]
            if len(others) < 2:
                continue
            n_o = len(others)
            j_idx, k_idx = np.triu_indices(n_o, k=1)
            xj = others[j_idx, 0]
            yj = others[j_idx, 1]
            xk = others[k_idx, 0]
            yk = others[k_idx, 1]
            
            xi, yi = pi
            areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            non_zero_areas = areas[areas > 0]
            if len(non_zero_areas) > 0:
                counts.update(non_zero_areas.tolist())
                
        if not counts:
            return Counter(), 0, 0
            
        best_area, best_count = counts.most_common(1)[0]
        return counts, int(best_area), int(best_count)

    def get_new_max_count_fast(counts, old_areas, new_areas, current_max_count, unchanged_max_cache):
        """
        Fast computation of new max count after replacing old_areas with new_areas.
        unchanged_max_cache: (max_count, max_area) for areas not in the changed set.
        """
        # Build delta for changed areas
        changed = set()
        for a in old_areas:
            if a > 0:
                changed.add(a)
        for a in new_areas:
            if a > 0:
                changed.add(a)
        
        # Compute delta counts
        delta = {}
        for a in old_areas:
            if a > 0:
                delta[a] = delta.get(a, 0) - 1
        for a in new_areas:
            if a > 0:
                delta[a] = delta.get(a, 0) + 1
        
        max_count = 0
        max_area = 0
        
        # Check changed areas
        for a in changed:
            if a == 0:
                continue
            new_c = counts.get(a, 0) + delta.get(a, 0)
            if new_c > max_count:
                max_count = new_c
                max_area = a
        
        # Check unchanged areas using cache
        if unchanged_max_cache is not None:
            cache_count, cache_area = unchanged_max_cache
            if cache_count > max_count:
                max_count = cache_count
                max_area = cache_area
        
        return max_count, max_area

    def compute_unchanged_max(counts, changed_areas):
        """Compute max count among areas not in changed_areas."""
        max_c = 0
        max_a = 0
        for a, c in counts.items():
            if a not in changed_areas and c > max_c:
                max_c = c
                max_a = a
        return (max_c, max_a)

    # --- Initialization with multiple patterns ---
    
    candidates = []
    
    # Pattern 1: Two parallel rows
    for y0 in range(-15, 15):
        for gap in [1, 2, 3, 4, 5, 6, 7]:
            y1 = y0 + gap
            if -30 <= y1 <= 30:
                for x_start in range(-15, 15):
                    pts = []
                    valid = True
                    for x in range(x_start, x_start + 15):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                            pts.append((x, y1))
                        else:
                            valid = False
                            break
                    if valid and len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)
                        
    # Pattern 2: Three parallel rows
    for y0 in range(-15, 15):
        for gap in [1, 2, 3]:
            y1 = y0 + gap
            y2 = y0 + 2 * gap
            if -30 <= y2 <= 30:
                for x_start in range(-15, 15):
                    pts = []
                    valid = True
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, y0))
                            pts.append((x, y1))
                            pts.append((x, y2))
                        else:
                            valid = False
                            break
                    if valid and len(pts) == 30 and len(set(pts)) == 30:
                        candidates.append(pts)

    # Pattern 3: Grids (various shapes)
    for w in range(4, 10):
        for h in range(4, 10):
            if w * h >= 30:
                for ox in range(-15, 15):
                    for oy in range(-15, 15):
                        pts = []
                        valid = True
                        for x in range(w):
                            for y in range(h):
                                px = ox + x
                                py = oy + y
                                if -30 <= px <= 30 and -30 <= py <= 30:
                                    pts.append((px, py))
                                else:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(pts) >= 30:
                            pts = pts[:30]
                            if len(set(pts)) == 30:
                                candidates.append(pts)

    # Pattern 4: Concentric-like clusters
    random.seed(42)
    for trial in range(20):
        cx = random.randint(-20, 20)
        cy = random.randint(-20, 20)
        pts = set()
        # Add points in a dense cluster
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if len(pts) >= 30:
                    break
                px, py = cx + dx, cy + dy
                if -30 <= px <= 30 and -30 <= py <= 30:
                    pts.add((px, py))
            if len(pts) >= 30:
                break
        if len(pts) == 30:
            candidates.append(list(pts))

    if not candidates:
        candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    # Evaluate initial candidates
    best_score = 0
    best_points = None
    best_counts = None
    best_area = 0
    
    for cand in candidates:
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        counts, area, count = compute_all_areas(cand)
        if count > best_score:
            best_score = count
            best_points = cand[:]
            best_counts = counts
            best_area = area

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_counts, best_area, best_score = compute_all_areas(best_points)

    # --- Local Search with multiple restarts ---
    
    all_valid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    def local_search(initial_points, initial_counts, initial_best_area, initial_best_count, time_budget):
        """Run local search from initial state for time_budget seconds."""
        current_counts = initial_counts.copy()
        current_best_area = initial_best_area
        current_best_count = initial_best_count
        current_points = initial_points[:]
        
        point_set = set(current_points)
        
        # Precompute unchanged max cache
        all_areas = set(current_counts.keys())
        unchanged_max_cache = compute_unchanged_max(current_counts, set())
        
        no_improvement_counter = 0
        max_no_improvement = 100
        
        while time.time() - start_time < DEADLINE and time.time() - (start_time + time_budget) < 0:
            improved = False
            
            point_order = list(range(30))
            random.shuffle(point_order)
            
            for i in point_order:
                if time.time() - start_time > DEADLINE:
                    break
                
                old_point = current_points[i]
                
                # Precompute 'others' data
                others_idx = [j for j in range(30) if j != i]
                others_pts = np.array([current_points[j] for j in others_idx], dtype=np.int64)
                
                n_o = len(others_pts)
                if n_o < 2:
                    continue
                    
                j_idx, k_idx = np.triu_indices(n_o, k=1)
                xj = others_pts[j_idx, 0]
                yj = others_pts[j_idx, 1]
                xk = others_pts[k_idx, 0]
                yk = others_pts[k_idx, 1]
                
                # Compute old areas for point i
                x0_old, y0_old = old_point
                old_areas = np.abs(x0_old * (yj - yk) + xj * (yk - y0_old) + xk * (y0_old - yj))
                
                # Generate candidates
                candidates_to_try = []
                
                # 1. Local perturbations (radius 3 for speed)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx = old_point[0] + dx
                        ny = old_point[1] + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            np_ = (nx, ny)
                            if np_ != old_point and np_ not in point_set:
                                candidates_to_try.append(np_)
                                
                # 2. Random valid points
                for _ in range(5):
                    np_ = random.choice(all_valid_points)
                    if np_ != old_point and np_ not in point_set:
                        candidates_to_try.append(np_)
                
                random.shuffle(candidates_to_try)
                
                for new_point in candidates_to_try[:15]:
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    x0, y0 = new_point
                    new_areas = np.abs(x0 * (yj - yk) + xj * (yk - y0) + xk * (y0 - yj))
                    
                    # Compute new max count
                    changed = set()
                    for a in old_areas:
                        if a > 0:
                            changed.add(a)
                    for a in new_areas:
                        if a > 0:
                            changed.add(a)
                    
                    # Use cache for unchanged max
                    new_max_count, new_max_area = get_new_max_count_fast(
                        current_counts, old_areas, new_areas, current_best_count, unchanged_max_cache
                    )
                    
                    if new_max_count > current_best_count:
                        # Accept move
                        # Update counts
                        for a in old_areas:
                            if a > 0:
                                current_counts[a] -= 1
                                if current_counts[a] == 0:
                                    del current_counts[a]
                        for a in new_areas:
                            if a > 0:
                                current_counts[a] = current_counts.get(a, 0) + 1
                                
                        # Update state
                        point_set.remove(old_point)
                        point_set.add(new_point)
                        current_points[i] = new_point
                        current_best_count = new_max_count
                        current_best_area = new_max_area
                        
                        # Update unchanged max cache
                        unchanged_max_cache = compute_unchanged_max(current_counts, changed)
                        
                        improved = True
                        no_improvement_counter = 0
                        break # Move to next point
            
            if not improved:
                no_improvement_counter += 1
                if no_improvement_counter > max_no_improvement:
                    break

        return current_points, current_counts, current_best_area, current_best_count

    # Run local search with time budget
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 5:
        # Single long run
        best_points, best_counts, best_area, best_score = local_search(
            best_points, best_counts, best_area, best_score, remaining
        )
    elif remaining > 2:
        # Shorter run
        best_points, best_counts, best_area, best_score = local_search(
            best_points, best_counts, best_area, best_score, remaining
        )

    return best_points
