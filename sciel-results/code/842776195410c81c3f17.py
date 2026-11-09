# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610  # Safety margin
    
    n = 33
    
    # Helper: Check if 4 points form a convex quadrilateral
    # Returns True if they are in convex position (none inside triangle of others)
    # Inputs must be distinct.
    def is_convex_quad(p0, p1, p2, p3):
        # Check if p0 is inside triangle p1-p2-p3
        # We can use area signs.
        # If all cross products have same sign, it's convex.
        # But we need to handle the specific ordering.
        # Actually, a set of 4 points is in convex position iff no point is inside the triangle formed by the other three.
        
        def area2(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            
        a1 = area2(p1, p2, p3)
        a2 = area2(p0, p2, p3)
        a3 = area2(p0, p1, p3)
        a4 = area2(p0, p1, p2)
        
        # If p0 is inside 123, then a1 (ref) and a2, a3, a4 signs?
        # Standard check: 
        # p0 inside 123 iff (a1 > 0 and a2 > 0 and a3 > 0) or (a1 < 0 and a2 < 0 and a3 < 0)
        # Note: a4 is area of 012.
        
        # Let's use a more robust method: 
        # The 4 points are in convex position iff the sum of areas of the 3 triangles formed by any 3 points 
        # is equal to the area of the convex hull? No, that's for 3 points.
        
        # Simpler: Check if any point is inside the triangle of the other 3.
        def point_in_triangle(pt, v1, v2, v3):
            # Returns True if pt is strictly inside v1-v2-v3
            # Using barycentric or cross product signs
            d1 = area2(pt, v1, v2)
            d2 = area2(pt, v2, v3)
            d3 = area2(pt, v3, v1)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            return not (has_neg and has_pos)

        if point_in_triangle(p0, p1, p2, p3): return False
        if point_in_triangle(p1, p0, p2, p3): return False
        if point_in_triangle(p2, p0, p1, p3): return False
        if point_in_triangle(p3, p0, p1, p2): return False
        
        return True

    def count_convex_7goms(points):
        """
        Count convex 7-gons using the property that a 7-gon is convex iff all its 4-point subsets are convex.
        C = sum_{S in C(n,4), S is convex} C(n-4, 3)
        """
        pts = points
        # Precompute convexity of all 4-subsets
        # 4-subset is convex if no point is inside the triangle of the other 3.
        
        n_subsets = 0
        # We need to iterate over all 4-combinations.
        # C(33,4) = 40920.
        # For each, check if convex.
        # If convex, add C(29,3) = 3654.
        
        # Optimization: Vectorize the point-in-triangle check?
        # It's hard to fully vectorize the "any of 4 points inside triangle" logic easily without complex masks,
        # but 40920 iterations is fast enough in Python if the inner check is tight.
        
        # Let's optimize the inner check.
        # For 4 points A, B, C, D:
        # They are convex iff:
        # (cross(B-A, C-A) * cross(B-A, D-A) > 0) AND (cross(C-B, D-B) * cross(C-B, A-B) > 0) ... no, this depends on ordering.
        
        # Robust check:
        # Sort 4 points by x, then y.
        # If they are convex, the two "middle" points in x-sort must be the diagonals?
        # Actually, the simplest robust check for 4 points:
        # Calculate areas of 4 triangles: ABC, ABD, ACD, BCD.
        # If the largest area is the sum of the other three (within epsilon), one point is inside.
        # If no such relation, it's convex.
        
        # Let's use the area sum check.
        # Area(ABC) + Area(ABD) + Area(ACD) + Area(BCD) ?
        # If D is inside ABC: Area(ABC) = Area(ABD) + Area(ACD) + Area(BCD).
        
        def area2(p1, p2, p3):
            return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

        total_count = 0
        # Iterate 4-combinations
        # To speed up, we can just loop.
        for i in range(n-3):
            p0 = pts[i]
            for j in range(i+1, n-2):
                p1 = pts[j]
                for k in range(j+1, n-1):
                    p2 = pts[k]
                    for l in range(k+1, n):
                        p3 = pts[l]
                        
                        a0 = area2(p0, p1, p2)
                        a1 = area2(p0, p1, p3)
                        a2 = area2(p0, p2, p3)
                        a3 = area2(p1, p2, p3)
                        
                        # Check if any point is inside the triangle of the other 3
                        # p0 inside 123? a3 == a0+a1+a2 ? (Approx)
                        # p1 inside 023? a2 == a0+a1+a3 ?
                        # p2 inside 013? a1 == a0+a2+a3 ?
                        # p3 inside 012? a0 == a1+a2+a3 ?
                        
                        # Due to floating point, use tolerance
                        tol = 1e-9
                        is_convex = True
                        if abs(a3 - (a0 + a1 + a2)) < tol: is_convex = False
                        elif abs(a2 - (a0 + a1 + a3)) < tol: is_convex = False
                        elif abs(a1 - (a0 + a2 + a3)) < tol: is_convex = False
                        elif abs(a0 - (a1 + a2 + a3)) < tol: is_convex = False
                        
                        if is_convex:
                            total_count += 3654 # C(29,3)
                            
        return total_count

    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        # Vectorized distance check
        # dist_sq[i,j]
        # We can do this efficiently
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Collinearity
        # Check area of triangles
        # 33C3 = 5456 checks
        for i in range(33):
            for j in range(i+1, 33):
                for l in range(j+1, 33):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[l]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area < 1e-9:
                        return False
        return True

    # Initial Construction
    # Use a "moment curve" projection or similar to minimize convex hulls?
    # Actually, points on a convex curve (like a parabola) have ALL subsets convex. That's bad (max C).
    # We want points to be "clustered" or have many points inside hulls of others.
    # A grid is good because many points are inside triangles/quads of others.
    
    # Start with a grid
    random.seed(42)
    np.random.seed(42)
    
    # 6x6 grid, remove 3 points, jitter
    grid_pts = []
    for i in range(6):
        for j in range(6):
            grid_pts.append((i/5.0, j/5.0))
    random.shuffle(grid_pts)
    grid_pts = grid_pts[:33]
    pts = np.array(grid_pts, dtype=float)
    pts += np.random.uniform(-0.02, 0.02, size=(33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    if not is_valid(pts):
        # Fallback to random if grid fails validity (unlikely with jitter)
        pts = np.random.rand(33, 2) * 0.98 + 0.01

    current_count = count_convex_7goms(pts)
    best_points = pts.copy()
    best_count = current_count
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    step_size = 0.05
    
    # Precompute C(29,3) = 3654
    # We can speed up counting by only checking 4-subsets that include the moved point?
    # Yes! If we move point i, only 4-subsets containing i can change convexity status.
    # Number of such subsets: C(32,3) = 4960.
    # This is much faster than 40920.
    
    def count_convex_7goms_delta(points, moved_idx, old_points):
        """
        Calculate change in count when moving point at moved_idx.
        Only 4-subsets containing moved_idx can change status.
        """
        # Subsets containing moved_idx: choose 3 from remaining 32.
        # C(32,3) = 4960.
        # For each, check if it was convex before and if it is convex now.
        # If was convex and now not: -3654
        # If was not and now convex: +3654
        
        # We need the old point coords
        old_p = old_points[moved_idx]
        new_p = points[moved_idx]
        
        delta = 0
        # Iterate 3-combinations of other indices
        others = [i for i in range(33) if i != moved_idx]
        
        # To make it faster, we can just re-evaluate the 4960 subsets.
        # But we need to know if they were convex before.
        # We can just check convexity for old and new.
        
        for i in range(len(others)):
            p1 = old_points[others[i]]
            for j in range(i+1, len(others)):
                p2 = old_points[others[j]]
                for k in range(j+1, len(others)):
                    p3 = old_points[others[k]]
                    
                    # Check old
                    old_convex = is_convex_quad(old_p, p1, p2, p3)
                    # Check new
                    new_convex = is_convex_quad(new_p, p1, p2, p3)
                    
                    if old_convex and not new_convex:
                        delta -= 3654
                    elif not old_convex and new_convex:
                        delta += 3654
                        
        return delta

    # Main SA Loop
    while time.time() < deadline:
        # Move a random point
        idx = random.randint(0, 32)
        old_p = best_points[idx].copy()
        
        # Generate new position
        # Gaussian perturbation
        new_p = old_p + np.random.normal(0, step_size, 2)
        new_p = np.clip(new_p, 0.0, 1.0)
        
        # Check validity locally (distance and collinearity with other points)
        # Full is_valid is expensive. Let's do a quick check.
        # Distance to all others
        valid = True
        for i in range(33):
            if i == idx: continue
            dx = new_p[0] - best_points[i, 0]
            dy = new_p[1] - best_points[i, 1]
            if dx*dx + dy*dy < 1e-12:
                valid = False
                break
        if not valid:
            continue
            
        # Collinearity check with pairs of others
        # This is expensive too. Maybe skip in SA and check at the end?
        # Risky. Let's do a sparse check or accept the cost.
        # Actually, 33*32/2 pairs = 528. For each, check against new_p?
        # No, collinearity involves 3 points.
        # We need to check if new_p is collinear with any pair (i, j).
        # C(32,2) = 496 pairs.
        for i in range(33):
            if i == idx: continue
            for j in range(i+1, 33):
                if j == idx: continue
                x1, y1 = best_points[i]
                x2, y2 = best_points[j]
                # Area of triangle (new_p, p_i, p_j)
                area = abs(new_p[0]*(y1-y2) + x1*(y2-new_p[1]) + x2*(new_p[1]-y1))
                if area < 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            continue

        # Calculate delta
        points_copy = best_points.copy()
        points_copy[idx] = new_p
        delta = count_convex_7goms_delta(points_copy, idx, best_points)
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            best_points[idx] = new_p
            current_count += delta
            if current_count < best_count:
                best_count = current_count
                # Optional: save best
            # Cool down
            T = max(T_min, T * cooling_rate)
            # Occasionally increase step size?
            if random.random() < 0.01:
                step_size = min(0.1, step_size * 1.05)
        else:
            # Reject
            T = max(T_min, T * cooling_rate)
            
        # Safety check every 1000 iterations
        # If we've been running a while, ensure we return something valid
        if time.time() > deadline - 10:
            break

    # Final validation
    if not is_valid(best_points):
        # Fallback
        return np.random.rand(33, 2) * 0.98 + 0.01
        
    return best_points

# EVOLVE-BLOCK-END
