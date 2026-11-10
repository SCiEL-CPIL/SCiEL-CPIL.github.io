import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    
    # Pre-compute all valid points in [-30, 30]^2
    # There are 61*61 = 3721 points.
    all_points = np.array([(x, y) for x in range(-30, 31) for y in range(-30, 31)], dtype=np.int64)
    
    # Helper: Compute all pair vectors and lengths? No, we need triangle areas.
    # To speed up incremental scoring, we can precompute areas between every pair of points?
    # That's 3721^2 * 3 ~ 40M operations. Might be too slow/memory heavy if not careful.
    # Let's stick to on-the-fly vectorized computation for the candidate set vs the new point.
    
    def score_set(pts):
        """
        Computes the max multiplicity of any positive doubled area for a set of points.
        pts: list of tuples or numpy array of shape (N, 2)
        """
        if len(pts) < 3:
            return 0, 0
        arr = np.array(pts, dtype=np.int64)
        counts = defaultdict(int)
        n = len(arr)
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                # Determinant: xi(yj-yk) + xj(yk-yi) + xk(yi-yj)
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    # Count occurrences of each area
                    # Using a dictionary update is safer for unique values
                    # But for speed, if we assume small areas, we can bin them
                    # However, areas can be up to ~1800. 
                    # Let's just use a counter on the unique values
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        if not counts:
            return 0, 0
        max_count = max(counts.values())
        max_area = [k for k, v in counts.items() if v == max_count][0]
        return max_count, max_area

    def incremental_gain(new_pt, current_pts, target_area):
        """
        Returns the number of triangles with exactly 'target_area' formed by 
        new_pt and any pair in current_pts.
        """
        if len(current_pts) < 2:
            return 0
        arr = np.array(current_pts, dtype=np.int64)
        nx, ny = new_pt
        # Vectors from new_pt to all current points
        # v_i = p_i - new_pt
        # Area( new_pt, p_i, p_j ) = 0.5 * | (p_i - new_pt) x (p_j - new_pt) |
        # Doubled Area = | (p_i[0]-nx)*(p_j[1]-ny) - (p_i[1]-ny)*(p_j[0]-nx) |
        
        # Vectorized cross product calculation
        # dx_i = arr[:, 0] - nx
        # dy_i = arr[:, 1] - ny
        # Cross_ij = dx_i * dy_j - dy_i * dx_j
        # We need |Cross_ij| == target_area
        
        dx = arr[:, 0] - nx
        dy = arr[:, 1] - ny
        
        # Compute cross product matrix C where C[i,j] = dx[i]*dy[j] - dy[i]*dx[j]
        # This is an outer product difference
        # C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
        C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
        
        # Count where |C[i,j]| == target_area
        # Note: Triangle (i, j) is same as (j, i), so we count pairs i < j
        # Also C[i,i] = 0, which is not target_area (target > 0)
        mask = np.abs(C) == target_area
        
        # We only want i < j to avoid double counting and self-loops
        # np.triu_indices
        i_idx, j_idx = np.triu_indices(len(arr), k=1)
        valid_pairs = mask[i_idx, j_idx]
        return np.sum(valid_pairs)

    best_overall_score = 0
    best_overall_pts = []
    
    # Strategy:
    # 1. Generate diverse seed sets (grids, lines, random).
    # 2. For each seed, identify the most frequent area A in the seed.
    # 3. Greedily add points that maximize the count of triangles with area A.
    # 4. Periodically re-evaluate the actual max multiplicity in case a different area becomes dominant.
    
    # Generate Seeds
    seeds = []
    
    # Regular Grids
    for w in [5, 6, 10, 15, 30]:
        h = 30 // w
        if h < 2: continue
        for dx in [1, 2, 3, 4, 5]:
            for dy in [1, 2, 3, 4, 5]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60: continue
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
    
    # Random Sets
    random.seed(42)
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        seeds.append(list(pts))
    
    # Deduplicate seeds
    unique_seeds = []
    seen = set()
    for s in seeds:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
    
    # Process Seeds
    for seed in unique_seeds:
        if time.time() > total_deadline - 5:
            break
            
        current_pts = list(seed)
        current_set = set(current_pts)
        
        # Initial Score
        cur_score, cur_area = score_set(current_pts)
        if cur_score > best_overall_score:
            best_overall_score = cur_score
            best_overall_pts = current_pts
            
        # Greedy Expansion
        # We will try to expand this set to 30 points (if not already) or refine it.
        # Since seeds are already size 30, we do a "refinement" phase:
        # Swap points to increase the count of the dominant area.
        
        target_area = cur_area if cur_area > 0 else 1
        
        # If the seed is already 30 points, we perform local swaps.
        # If we want to be smarter, we can try to replace points that contribute little to target_area
        # with points that contribute a lot.
        
        improved = True
        max_iterations = 50 # Limit swaps per seed to save time
        
        while improved and max_iterations > 0 and time.time() < total_deadline - 2:
            improved = False
            max_iterations -= 1
            
            # Identify the "worst" point: one that is part of fewest triangles with target_area
            # Or simply try swapping each point with a random candidate?
            # Let's try swapping each point with a few candidates.
            
            # To be efficient, we can precompute the contribution of each point to the current set's target_area triangles.
            # Contribution of point p_i = number of pairs (p_j, p_k) in current_pts\{p_i} such that Area(p_i, p_j, p_k) == target_area
            
            # This is expensive to recompute fully each step.
            # Alternative: Random Swaps with Acceptance
            # Pick random i, random new_p. Calculate delta in score.
            
            # Let's stick to the "Greedy Add" logic if we started smaller, but we started at 30.
            # So we use "Greedy Swap".
            
            # To speed up, we only evaluate a subset of potential new points?
            # No, we should pick the best new point from a random sample of 100 candidates to replace the worst point.
            
            # 1. Find worst point index
            # We approximate worst point by picking a random point and seeing if we can improve by replacing it.
            # Actually, let's just do random swaps with hill climbing.
            
            idx = random.randint(0, 29)
            old_p = current_pts[idx]
            
            # Generate candidate replacements
            # Sample 20 random points from the grid
            candidates = random.sample(range(len(all_points)), 20)
            
            best_candidate = None
            best_candidate_gain = 0 # Delta in number of target_area triangles
            
            # Calculate current count of target_area triangles in the full set
            # This is expensive. 
            # Instead, let's calculate the gain of replacing old_p with new_p.
            # Gain = Count(new_p, others) - Count(old_p, others)
            # Where Count(p, others) is number of triangles with area target_area involving p and two others from the set.
            
            # Let's compute Count(old_p, others) once.
            others = [p for i, p in enumerate(current_pts) if i != idx]
            
            def count_involving(pt, others_list, t_area):
                if len(others_list) < 2: return 0
                arr_o = np.array(others_list, dtype=np.int64)
                nx, ny = pt
                dx = arr_o[:, 0] - nx
                dy = arr_o[:, 1] - ny
                C = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
                mask = np.abs(C) == t_area
                i_idx, j_idx = np.triu_indices(len(arr_o), k=1)
                return np.sum(mask[i_idx, j_idx])
            
            old_count = count_involving(old_p, others, target_area)
            
            for c_idx in candidates:
                new_p = tuple(all_points[c_idx])
                if new_p in current_set:
                    continue
                new_count = count_involving(new_p, others, target_area)
                gain = new_count - old_count
                if gain > best_candidate_gain:
                    best_candidate_gain = gain
                    best_candidate = new_p
            
            if best_candidate is not None:
                # Perform swap
                current_pts[idx] = best_candidate
                current_set.remove(old_p)
                current_set.add(best_candidate)
                improved = True
                
                # Re-evaluate full score periodically or if gain was significant
                # Since we only optimized for one area, the max area might have shifted.
                # Let's re-score every few improvements.
                if max_iterations % 5 == 0 or max_iterations == 0:
                    new_score, new_area = score_set(current_pts)
                    if new_score > best_overall_score:
                        best_overall_score = new_score
                        best_overall_pts = current_pts
                    # Update target area if it changed
                    if new_area > 0:
                        target_area = new_area

    return best_overall_pts
