import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    total_deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Pre-compute all valid points in [-30, 30]^2
    all_points = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
    num_points = len(all_points)
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}
    
    # Helper: Compute histogram of doubled areas for a set of 30 points
    def get_area_histogram(pts):
        counts = defaultdict(int)
        n = len(pts)
        arr = np.array(pts, dtype=np.int64)
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                dx1 = xj - xi
                dy1 = yj - yi
                dx2 = rest[:, 0] - xi
                dy2 = rest[:, 1] - yi
                doubled_areas = np.abs(dx1 * dy2 - dy1 * dx2)
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        return counts

    def get_max_score(hist):
        if not hist:
            return 0, 0
        max_count = max(hist.values())
        # Pick the smallest area in case of tie for stability
        max_area = min([k for k, v in hist.items() if v == max_count])
        return max_count, max_area

    # Helper: Compute the delta in histogram when swapping old_pt with new_pt
    # given the other 28 points
    def compute_delta_hist(old_pt, new_pt, others_arr):
        deltas = defaultdict(int)
        if len(others_arr) < 2: return deltas
        
        def get_involving_areas(pt, others_arr):
            ox, oy = pt
            dx = others_arr[:, 0] - ox
            dy = others_arr[:, 1] - oy
            # Cross product with all pairs in others
            # C[i, j] = dx[i]*dy[j] - dy[i]*dx[j]
            # We need i < j in others
            C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
            abs_C = np.abs(C)
            i_idx, j_idx = np.triu_indices(len(others_arr), k=1)
            valid_areas = abs_C[i_idx, j_idx]
            valid_areas = valid_areas[valid_areas > 0]
            return valid_areas

        old_areas = get_involving_areas(old_pt, others_arr)
        if len(old_areas) > 0:
            vals, cnts = np.unique(old_areas, return_counts=True)
            for v, c in zip(vals, cnts):
                deltas[int(v)] -= int(c)
        
        new_areas = get_involving_areas(new_pt, others_arr)
        if len(new_areas) > 0:
            vals, cnts = np.unique(new_areas, return_counts=True)
            for v, c in zip(vals, cnts):
                deltas[int(v)] += int(c)
        
        return deltas

    # Generate diverse seeds
    seeds = []
    
    # 1. Grids
    for w in range(4, 10):
        for h in range(4, 10):
            if w * h < 30: continue
            for dx in [2, 3, 4, 5]:
                for dy in [2, 3, 4, 5]:
                    for ox in range(-10, 11, 5):
                        for oy in range(-10, 11, 5):
                            pts = []
                            valid = True
                            for i in range(w):
                                for j in range(h):
                                    x = ox + i * dx
                                    y = oy + j * dy
                                    if -30 <= x <= 30 and -30 <= y <= 30:
                                        pts.append((x, y))
                                    else:
                                        valid = False
                                        break
                                if not valid: break
                            if valid and len(pts) == 30:
                                seeds.append(pts)
    
    # 2. Random
    random.seed(42)
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        seeds.append(list(pts))
        
    # Deduplicate
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    best_overall_score = 0
    best_overall_pts = None
    best_overall_hist = None
    
    num_seeds = len(unique_seeds)
    # Allocate time per seed
    time_per_seed = 30.0 / num_seeds if num_seeds > 0 else 30.0
    time_per_seed = max(0.5, min(2.0, time_per_seed))

    for seed in unique_seeds:
        if time.time() > total_deadline:
            break
            
        current_pts = list(seed)
        current_set = set(current_pts)
        
        cur_hist = get_area_histogram(current_pts)
        cur_score, cur_area = get_max_score(cur_hist)
        
        if cur_score > best_overall_score:
            best_overall_score = cur_score
            best_overall_pts = current_pts
            best_overall_hist = cur_hist.copy()
            
        target_area = cur_area
        improved = True
        iter_count = 0
        max_iterations = 200 # Fixed budget per seed
        
        while improved and iter_count < max_iterations and time.time() < total_deadline:
            improved = False
            iter_count += 1
            
            # Geometric Targeting Search
            # 1. Pick a random "anchor" point from the current set
            anchor_idx = random.randint(0, 29)
            anchor_pt = current_pts[anchor_idx]
            
            # 2. Find all other points to form base pairs with the anchor
            others = [current_pts[i] for i in range(30) if i != anchor_idx]
            others_arr = np.array(others, dtype=np.int64)
            
            # 3. For each base pair (anchor, other), find candidate 3rd points 
            #    that would create a triangle of `target_area`.
            #    Area = 0.5 * |det(P2-P1, P3-P1)| = target_area
            #    det(P2-P1, P3-P1) = +/- 2*target_area
            #    Let P1 = anchor, P2 = other.
            #    (P2x-P1x)*(P3y-P1y) - (P2y-P1y)*(P3x-P1x) = +/- 2*target_area
            #    This is a linear equation for P3 = (x,y):
            #    A*x + B*y = C
            #    where A = -(P2y-P1y), B = (P2x-P1x), C = 2*target_area + A*P1x + B*P1y
            
            # We will collect all valid integer points (x,y) in [-30,30]^2 that satisfy this.
            # To be efficient, we don't scan the whole grid for each pair.
            # Instead, we can just check the candidates against the grid points.
            # Since the grid is small (61x61), we can pre-filter.
            
            # Actually, a simpler approach:
            # For each base pair, generate the specific candidate points.
            # If A=0 and B=0, skip (collinear).
            # If A!=0 or B!=0, the solutions (x,y) are on a line.
            # We can iterate over the range of x or y and check if the other coordinate is integer and in range.
            
            candidate_points = set()
            
            for other_pt in others:
                ox, oy = other_pt
                ax, ay = anchor_pt
                
                dx = ox - ax
                dy = oy - ay
                
                # Equation: -dy * x + dx * y = 2*target_area + dy*ax - dx*ay
                # Let C_pos = 2*target_area + dy*ax - dx*ay
                # Let C_neg = -2*target_area + dy*ax - dx*ay
                # We need points (x,y) such that -dy*x + dx*y is C_pos or C_neg.
                
                # If dx == 0 and dy == 0, skip (duplicate point, shouldn't happen)
                if dx == 0 and dy == 0:
                    continue
                
                # We want to find integer solutions (x,y) in [-30, 30]
                # -dy * x + dx * y = C
                # If dx == 0: dy * y = C => y = C/dy. Check if integer and in range. x is free? No, x is constrained by the grid.
                # Actually, if dx=0, the line is horizontal y = const. Any x in range works.
                # If dy=0, the line is vertical x = const. Any y in range works.
                # If both non-zero, we can iterate over x and check y.
                
                # To keep it fast, we will just check a few specific candidates or iterate if the line is axis-aligned.
                # For generic lines, the number of integer points in a 61x61 box is small.
                
                # Optimization: We only care about points NOT in current_set.
                
                # Case 1: dx == 0
                if dx == 0:
                    # dy * y = C => y = C / dy
                    for C in [2*target_area + dy*ax, -2*target_area + dy*ax]:
                        if C % dy == 0:
                            y_val = C // dy
                            if -30 <= y_val <= 30:
                                # x can be any integer in [-30, 30]
                                # We can't add all of them, too many.
                                # We will sample a few or just add the anchor's x?
                                # Actually, for local search, we just need *some* good candidates.
                                # Let's add (ax, y_val) and maybe neighbors.
                                candidate_points.add((ax, y_val))
                                if ax > -30: candidate_points.add((ax-1, y_val))
                                if ax < 30: candidate_points.add((ax+1, y_val))
                
                # Case 2: dy == 0
                elif dy == 0:
                    # -dx * x = C => x = -C / dx
                    for C in [2*target_area - dx*ay, -2*target_area - dx*ay]:
                        if C % dx == 0:
                            x_val = -C // dx
                            if -30 <= x_val <= 30:
                                candidate_points.add((x_val, ay))
                                if ay > -30: candidate_points.add((x_val, ay-1))
                                if ay < 30: candidate_points.add((x_val, ay+1))
                
                # Case 3: Both non-zero
                else:
                    # -dy * x + dx * y = C
                    # y = (C + dy * x) / dx
                    # We iterate x in [-30, 30]
                    for C in [2*target_area + dy*ax - dx*ay, -2*target_area + dy*ax - dx*ay]:
                        for x_val in range(-30, 31):
                            num = C + dy * x_val
                            if num % dx == 0:
                                y_val = num // dx
                                if -30 <= y_val <= 30:
                                    candidate_points.add((x_val, y_val))
            
            # Filter candidates: must be valid, not in current_set
            valid_candidates = [p for p in candidate_points if p not in current_set]
            
            if not valid_candidates:
                # Fallback: random swap
                idx = random.randint(0, 29)
                old_p = current_pts[idx]
                while True:
                    new_p = tuple(all_points[random.randint(0, num_points-1)])
                    if new_p not in current_set:
                        break
                # Update
                others = [current_pts[i] for i in range(30) if i != idx]
                others_arr = np.array(others, dtype=np.int64)
                delta_hist = compute_delta_hist(old_p, new_p, others_arr)
                for area, d in delta_hist.items():
                    cur_hist[area] += d
                    if cur_hist[area] == 0:
                        del cur_hist[area]
                current_pts[idx] = new_p
                current_set.remove(old_p)
                current_set.add(new_p)
                improved = True
                continue

            # Evaluate candidates
            # We want to pick the candidate that maximizes the count of target_area triangles.
            # Since we generated them specifically to form target_area with the anchor,
            # each candidate adds at least 1 to the target_area count (for the triangle with anchor).
            # It might add more if it forms target_area with other points too.
            
            # To be safe and fast, we will just pick the first valid candidate that doesn't break distinctness
            # and assume it improves the score. We will verify by updating the histogram.
            
            # Pick a random candidate from the valid list to add diversity
            best_candidate = random.choice(valid_candidates)
            
            # Determine which point to replace.
            # Ideally, replace a point that contributes least to the target_area.
            # For simplicity, replace the anchor? No, anchor is fixed in the generation.
            # We should replace one of the points in `others` or the anchor.
            # If we replace the anchor, we lose the geometry we just calculated.
            # So we should replace one of the `others`.
            # Which one? The one that is "least useful".
            # Let's just pick a random index from `others`.
            
            # Find index of a random other
            other_to_replace = random.choice(others)
            replace_idx = current_pts.index(other_to_replace)
            old_p = current_pts[replace_idx]
            
            # Compute delta
            # Note: `others_arr` in compute_delta_hist should be the 28 points excluding the one being replaced.
            # The set of points is: Anchor + (Others - {other_to_replace}) + New_Point
            # The "others" for the delta calculation are the 28 points that stay.
            # Staying points = [current_pts[i] for i in range(30) if i != replace_idx]
            # This includes the Anchor and the other 27 points.
            
            staying_pts = [current_pts[i] for i in range(30) if i != replace_idx]
            staying_arr = np.array(staying_pts, dtype=np.int64)
            
            delta_hist = compute_delta_hist(old_p, best_candidate, staying_arr)
            
            # Apply delta
            for area, d in delta_hist.items():
                cur_hist[area] += d
                if cur_hist[area] == 0:
                    del cur_hist[area]
            
            # Update state
            current_pts[replace_idx] = best_candidate
            current_set.remove(old_p)
            current_set.add(best_candidate)
            improved = True
            
            # Update target area occasionally
            if iter_count % 5 == 0:
                new_score, new_area = get_max_score(cur_hist)
                if new_score > best_overall_score:
                    best_overall_score = new_score
                    best_overall_pts = current_pts.copy()
                    best_overall_hist = cur_hist.copy()
                if new_area != target_area:
                    target_area = new_area

    if best_overall_pts is None:
        pts = []
        for i in range(30):
            x = i % 6
            y = i // 6
            pts.append((x*5-15, y*5-10))
        best_overall_pts = pts
        
    return best_overall_pts
