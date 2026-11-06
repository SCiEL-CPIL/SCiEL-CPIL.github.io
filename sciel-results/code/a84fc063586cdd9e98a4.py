"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    def dist_sq(a1, b1, a2, b2):
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db

    def count_edges(points):
        n = len(points)
        count = 0
        for i in range(n):
            ai, bi = points[i]
            for j in range(i+1, n):
                if dist_sq(ai, bi, points[j][0], points[j][1]) == 1:
                    count += 1
        return count

    def hex_spiral(n_points):
        """Generate n_points in hexagonal spiral order on triangular lattice."""
        points = [(0, 0)]
        if n_points <= 1:
            return points[:n_points]
        
        r = 1
        while len(points) < n_points:
            # Ring r: 6r points
            # Side 1: (r,0) to (r-1,1) to (r-2,2)... direction (-1,1), r points
            for i in range(r):
                points.append((r - i, i))
                if len(points) >= n_points:
                    return points[:n_points]
            # Side 2: (0,r) to (-1,r) to (-2,r)... direction (-1,0), r-1 points
            for i in range(1, r):
                points.append((-i, r))
                if len(points) >= n_points:
                    return points[:n_points]
            # Side 3: (-r,r) to (-r,r-1) to (-r,r-2)... direction (0,-1), r-1 points
            for i in range(1, r):
                points.append((-r, r - i))
                if len(points) >= n_points:
                    return points[:n_points]
            # Side 4: (-r,0) to (-r+1,-1) to (-r+2,-2)... direction (1,-1), r-1 points
            for i in range(1, r):
                points.append((-r + i, -i))
                if len(points) >= n_points:
                    return points[:n_points]
            # Side 5: (0,-r) to (1,-r) to (2,-r)... direction (1,0), r-1 points
            for i in range(1, r):
                points.append((i, -r))
                if len(points) >= n_points:
                    return points[:n_points]
            # Side 6: (r,-r) to (r,-r+1) to (r,-r+2)... direction (0,1), r-1 points
            for i in range(1, r):
                points.append((r, -r + i))
                if len(points) >= n_points:
                    return points[:n_points]
            r += 1
        
        return points[:n_points]

    # Get initial hex spiral solution
    best_points = hex_spiral(32)
    best_score = count_edges(best_points)
    
    # Local search with point_set for efficiency
    point_set = set(tuple(p) for p in best_points)
    
    # Precompute neighbor offsets for unit distance
    # dist_sq(da,db)==1: da^2+da*db+db^2=1
    # Solutions: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
    unit_neighbors = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    
    # Determine bounding box for candidate positions
    a_vals = [p[0] for p in best_points]
    b_vals = [p[1] for p in best_points]
    a_min, a_max = min(a_vals) - 3, max(a_vals) + 3
    b_min, b_max = min(b_vals) - 3, max(b_vals) + 3
    
    # Generate candidate positions
    candidates = []
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            if abs(a) <= 64 and abs(b) <= 64:
                candidates.append((a, b))
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(32):
            if time.time() > deadline - 5:
                break
            a_i, b_i = best_points[i]
            old_pos = (a_i, b_i)
            
            # Try moving point i to each candidate
            for a_new, b_new in candidates:
                if (a_new, b_new) == old_pos:
                    continue
                if (a_new, b_new) in point_set:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for j in range(32):
                    if j == i:
                        continue
                    if dist_sq(a_new, b_new, best_points[j][0], best_points[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try the move
                point_set.remove(old_pos)
                point_set.add((a_new, b_new))
                best_points[i] = (a_new, b_new)
                
                new_score = count_edges(best_points)
                
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    point_set.remove((a_new, b_new))
                    point_set.add(old_pos)
                    best_points[i] = old_pos
            
            if best_score == 76:
                break
    
    # If we haven't reached 76, try a few more strategies
    if best_score < 76 and time.time() < deadline - 10:
        # Try swapping two points
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            i, j = random.sample(range(32), 2)
            pi, pj = best_points[i], best_points[j]
            point_set.remove(pi)
            point_set.remove(pj)
            best_points[i] = pj
            best_points[j] = pi
            point_set.add(pi)
            point_set.add(pj)
            
            new_score = count_edges(best_points)
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert
                best_points[i] = pi
                best_points[j] = pj
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
