import numpy as np
import itertools
import random

def construct():
    # Define the search space
    # We will use a pool of candidate points. 
    # A dense grid is a good pool.
    # Let's use points in [-10, 10] x [-10, 10] as a pool, 
    # but ensure we can pick 30 distinct ones.
    
    def get_pool():
        pts = []
        for x in range(-10, 11):
            for y in range(-10, 11):
                pts.append((x, y))
        return pts

    pool = get_pool()
    
    # Initial configuration: A 6x5 grid centered at 0
    # x in [-2, 3], y in [-2, 2] -> 6*5=30 points
    init_pts = []
    for x in range(-2, 4):
        for y in range(-2, 3):
            init_pts.append((x, y))
            
    current_pts = set(init_pts)
    
    # Helper to calculate count of area A in a set of points
    def count_area(points, target_area):
        count = 0
        n = len(points)
        # Use list for faster indexing if needed, but set is fine for membership
        # We need indices for triplets
        pts_list = list(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts_list[i]
                    x2, y2 = pts_list[j]
                    x3, y3 = pts_list[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det == target_area:
                        count += 1
        return count

    # Helper to calculate how many triangles of area A involve a specific point P 
    # given a set of other points S
    def count_involving(P, S, target_area):
        count = 0
        # S is a list of points
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = P
                x2, y2 = S[i]
                x3, y3 = S[j]
                det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if det == target_area:
                    count += 1
        return count

    # Local Search
    best_score = 0
    best_points = list(current_pts)
    
    # We will run the search for a few iterations
    # To speed up, we might use a smaller pool or optimized logic.
    # Given 60s, we can afford a few thousand iterations of this O(N^3) or O(N^2 * Pool) logic.
    
    # Let's optimize count_involving using NumPy if possible, 
    # but pure Python might be okay for 30 points.
    # 30 points -> 435 pairs. 
    # If we check 100 candidates, that's 43,500 determinant calcs.
    # In Python, this is fast (~0.01s).
    
    # We need to update the "best area" dynamically.
    
    for iteration in range(500):
        # 1. Determine current best area and score
        # Calculate all areas
        areas = []
        pts_list = list(current_pts)
        n = len(pts_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts_list[i]
                    x2, y2 = pts_list[j]
                    x3, y3 = pts_list[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        areas.append(det)
        
        if not areas:
            break
            
        from collections import Counter
        area_counts = Counter(areas)
        current_area, current_count = max(area_counts.items(), key=lambda x: x[1])
        
        if current_count > best_score:
            best_score = current_count
            best_points = list(current_pts)
            
        # 2. Find the point to remove
        # For each point, calculate how many triangles of current_area it forms
        # with the other 29 points.
        worst_point = None
        min_contribution = float('inf')
        
        for p in current_pts:
            others = [q for q in current_pts if q != p]
            contrib = count_involving(p, others, current_area)
            if contrib < min_contribution:
                min_contribution = contrib
                worst_point = p
                
        # 3. Find a better replacement
        # Try candidates from the pool that are not in current_pts
        # We don't need to try ALL, just a sample or a promising subset.
        # Let's try all candidates in the pool not in current_pts.
        # Pool size is 21*21 = 441. 
        # 441 * 435 ops = ~190k ops. Fast.
        
        candidates = [p for p in pool if p not in current_pts]
        
        best_candidate = None
        max_new_contribution = -1
        
        for c in candidates:
            # Calculate contribution of c with the set (current_pts - {worst_point})
            others = [q for q in current_pts if q != worst_point]
            contrib = count_involving(c, others, current_area)
            if contrib > max_new_contribution:
                max_new_contribution = contrib
                best_candidate = c
                
        # 4. Perform swap if improved
        if best_candidate is not None and max_new_contribution > min_contribution:
            current_pts.remove(worst_point)
            current_pts.add(best_candidate)
        else:
            # No improvement found with current_area
            # Maybe try a different area? 
            # For simplicity, we break or randomize.
            # To avoid getting stuck, we can try a random move if stuck for a while.
            # But for now, let's just break or do a random perturbation.
            # Let's try a random swap to escape local minima.
            if iteration % 50 == 0:
                # Random swap
                if len(current_pts) > 0:
                    p_to_remove = random.choice(list(current_pts))
                    c_to_add = random.choice(candidates)
                    current_pts.remove(p_to_remove)
                    current_pts.add(c_to_add)
            else:
                break
                
    return best_points
