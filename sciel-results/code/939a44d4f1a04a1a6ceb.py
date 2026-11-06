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

    def hex_dist(a, b):
        """Hexagonal distance from origin on triangular lattice."""
        return max(abs(a), abs(b), abs(a+b))

    def hex_spiral(n_points):
        """Generate n_points in hexagonal spiral order on triangular lattice.
        
        Uses the ring-by-ring approach where each ring r contains all points
        at hex distance exactly r, ordered by angle for spiral continuity.
        """
        points = [(0, 0)]
        if n_points <= 1:
            return points[:n_points]
        
        r = 1
        while len(points) < n_points:
            # Generate all points at hex distance == r
            ring_points = []
            for a in range(-r, r+1):
                for b in range(-r, r+1):
                    if hex_dist(a, b) == r:
                        ring_points.append((a, b))
            
            # Sort by angle to maintain spiral order
            # angle = atan2(b*sqrt(3)/2, a + b/2)
            def angle_key(p):
                a, b = p
                x = a + b * 0.5
                y = b * (math.sqrt(3) / 2)
                return math.atan2(y, x)
            
            ring_points.sort(key=angle_key)
            
            for p in ring_points:
                if len(points) >= n_points:
                    return points
                points.append(p)
            
            r += 1
        
        return points[:n_points]

    # Get initial hex spiral solution
    best_points = hex_spiral(32)
    best_score = count_edges(best_points)
    
    # If we already have 76, return immediately
    if best_score == 76:
        return [[a, b] for a, b in best_points]
    
    # Local search with efficient edge counting
    # Precompute unit neighbor offsets
    unit_neighbors = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    
    def count_edges_fast(points, changed_idx=None, old_pos=None):
        """Count edges, optionally skipping re-counting pairs not involving changed point."""
        if changed_idx is None:
            return count_edges(points)
        
        n = len(points)
        # Count all edges not involving changed_idx
        count = 0
        for i in range(n):
            if i == changed_idx:
                continue
            ai, bi = points[i]
            for j in range(i+1, n):
                if j == changed_idx:
                    continue
                if dist_sq(ai, bi, points[j][0], points[j][1]) == 1:
                    count += 1
        
        # Count edges involving changed_idx
        ai, bi = points[changed_idx]
        for j in range(n):
            if j == changed_idx:
                continue
            if dist_sq(ai, bi, points[j][0], points[j][1]) == 1:
                count += 1
        
        return count

    point_set = set(tuple(p) for p in best_points)
    
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
                
                new_score = count_edges_fast(best_points, i, old_pos)
                
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    if best_score == 76:
                        return [[a, b] for a, b in best_points]
                else:
                    # Revert
                    point_set.remove((a_new, b_new))
                    point_set.add(old_pos)
                    best_points[i] = old_pos
            
            if best_score == 76:
                break
    
    # If we haven't reached 76, try random restarts with different spiral orientations
    if best_score < 76 and time.time() < deadline - 10:
        # Try different starting points for the spiral
        for offset_a in range(-2, 3):
            for offset_b in range(-2, 3):
                if time.time() > deadline - 10:
                    break
                
                # Generate spiral centered at (offset_a, offset_b)
                points = []
                for a, b in best_points:
                    points.append((a + offset_a, b + offset_b))
                
                # Check validity
                valid = True
                for a, b in points:
                    if abs(a) > 64 or abs(b) > 64:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                score = count_edges(points)
                if score > best_score:
                    best_score = score
                    best_points = points
                    if best_score == 76:
                        return [[a, b] for a, b in best_points]
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
