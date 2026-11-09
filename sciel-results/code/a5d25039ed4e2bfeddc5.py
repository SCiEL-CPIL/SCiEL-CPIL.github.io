# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Minimize the number of convex 7-gons among 33 points in the unit square.
    Uses a structured initial construction plus local search to reduce C.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit
    n = 33
    k = 7
    
    def count_convex_7goms(points):
        """
        Count convex 7-point subsets using the dynamic programming approach described.
        Points sorted lexicographically. For each pivot i, later points sorted by atan2 angle.
        A turn p->q->r is convex if cross(q-p, r-q) > 1e-9.
        We count 7-cycles that are convex (all turns including closing are convex).
        """
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        total = 0
        npts = len(pts)
        
        for i in range(npts):
            pivot = pts[i]
            later = pts[i+1:]
            if len(later) < 6:
                continue
            # Sort later points by angle around pivot
            angles = []
            for j, p in enumerate(later):
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                ang = math.atan2(dy, dx)
                angles.append((ang, j, p))
            angles.sort(key=lambda x: x[0])
            
            # Now we have m = len(later) points sorted by angle around pivot
            # We need to find sequences of 6 points from these (in angular order, wrapping)
            # such that the 7-gon (pivot + 6 points) is convex.
            # The turns are: pivot -> a0 -> a1 -> ... -> a5 -> pivot
            # Each consecutive triple must have positive cross product.
            
            m = len(angles)
            if m < 6:
                continue
            
            # For a convex polygon containing pivot and 6 other points in angular order,
            # the 6 points must appear in circular order around pivot.
            # We need to select 6 points from the m angularly-sorted points such that
            # all turns are convex.
            
            # This is complex. Let's use a simpler approach:
            # For each combination of 6 points from the m later points,
            # check if they form a convex chain with the pivot.
            # But that's C(m,6) which can be large.
            
            # Alternative: since points are sorted by angle, a convex 7-gon with pivot
            # must use 6 points that are in angular order. The key insight is that if
            # the 6 points are in angular order around pivot, the polygon is convex iff
            # all the "turns" are consistent.
            
            # Let's just check all combinations of 6 from m.
            # m is at most 32, C(32,6) = 906192 which is too many per pivot.
            
            # Better approach: use the fact that in a convex polygon, the vertices
            # appear in order. So we need 6 points in angular order where the polygon
            # is convex. This can be done with DP.
            
            # Actually, let me use a different strategy: just check all C(32,6) per pivot
            # but only for pivots where m is small, or use a smarter method.
            
            # For now, let's use a simpler counting method that's faster:
            # A set of 7 points is in convex position iff all 7 are on the convex hull.
            # So we can just check if the 7 points form a convex polygon.
            
            # But that's still C(33,7) = 42921450 checks total, which might be too slow.
            
            # Let me use the angular sorting approach more carefully.
            # For fixed pivot i, the 6 other points must be in angular order.
            # So we need to find 6-element subsets of {0,...,m-1} (indices in angles)
            # such that the 7-gon is convex.
            
            # The turns to check:
            # pivot -> angles[a0] -> angles[a1] -> ... -> angles[a5] -> pivot
            # Cross products must all be > 1e-9.
            
            # This is equivalent to: the 6 points form a convex chain when viewed
            # from the pivot, and the first and last connect back to pivot convexly.
            
            # Let me just do the brute force for now but with early termination.
            
            for combo in combinations(range(m), 6):
                a0, a1, a2, a3, a4, a5 = combo
                p0 = angles[a0][2]
                p1 = angles[a1][2]
                p2 = angles[a2][2]
                p3 = angles[a3][2]
                p4 = angles[a4][2]
                p5 = angles[a5][2]
                
                # Check turn pivot -> p0 -> p1
                cx = (p0[0] - pivot[0]) * (p1[1] - p0[1]) - (p0[1] - pivot[1]) * (p1[0] - p0[0])
                if cx <= 1e-9:
                    continue
                # p1 -> p2
                cx = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
                if cx <= 1e-9:
                    continue
                # p2 -> p3
                cx = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
                if cx <= 1e-9:
                    continue
                # p3 -> p4
                cx = (p3[0] - p2[0]) * (p4[1] - p3[1]) - (p3[1] - p2[1]) * (p4[0] - p3[0])
                if cx <= 1e-9:
                    continue
                # p4 -> p5
                cx = (p4[0] - p3[0]) * (p5[1] - p4[1]) - (p4[1] - p3[1]) * (p5[0] - p4[0])
                if cx <= 1e-9:
                    continue
                # p5 -> pivot
                cx = (p5[0] - p4[0]) * (pivot[1] - p5[1]) - (p5[1] - p4[1]) * (pivot[0] - p5[0])
                if cx <= 1e-9:
                    continue
                # pivot -> p0 (closing turn)
                cx = (pivot[0] - p5[0]) * (p0[1] - pivot[1]) - (pivot[1] - p5[1]) * (p0[0] - pivot[0])
                if cx <= 1e-9:
                    continue
                
                total += 1
        
        return total
    
    def is_valid(points):
        """Check validity constraints."""
        if len(points) != 33:
            return False
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Check pairwise distances
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Check no three collinear
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
    
    # Initial construction: use a grid-like pattern with perturbations
    # to avoid collinearity and create a structured arrangement.
    
    def generate_initial():
        """Generate an initial set of 33 points."""
        random.seed(42)
        np.random.seed(42)
        
        # Try a few different initial constructions and pick the best
        candidates = []
        
        # Construction 1: Random points
        pts1 = np.random.rand(33, 2)
        candidates.append(pts1)
        
        # Construction 2: Grid with jitter
        # 6x6 grid = 36 points, remove 3, add jitter
        grid_pts = []
        for i in range(6):
            for j in range(6):
                grid_pts.append((i/5.0, j/5.0))
        # Remove 3 random points
        random.shuffle(grid_pts)
        grid_pts = grid_pts[:33]
        # Add small jitter
        pts2 = np.array(grid_pts, dtype=float)
        pts2 += np.random.uniform(-0.01, 0.01, size=(33, 2))
        pts2 = np.clip(pts2, 0, 1)
        candidates.append(pts2)
        
        # Construction 3: Points on a circle-like curve
        theta = np.linspace(0, 2*np.pi, 33, endpoint=False)
        r = 0.45 + 0.05 * np.sin(3*theta)
        pts3 = np.column_stack([0.5 + r*np.cos(theta), 0.5 + r*np.sin(theta)])
        candidates.append(pts3)
        
        # Pick the best candidate
        best_pts = None
        best_count = float('inf')
        for cand in candidates:
            if not is_valid(cand):
                # Try to fix
                cand = cand + np.random.uniform(-0.001, 0.001, size=cand.shape)
                cand = np.clip(cand, 0, 1)
                if not is_valid(cand):
                    continue
            cnt = count_convex_7goms(cand)
            if cnt < best_count:
                best_count = cnt
                best_pts = cand.copy()
        
        if best_pts is None:
            best_pts = candidates[0]
        
        return best_pts, best_count
    
    best_points, best_count = generate_initial()
    
    # Local search: perturb random points and accept if count decreases
    random.seed(123)
    np.random.seed(123)
    
    current_points = best_points.copy()
    current_count = best_count
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Perturb a random point
        idx = random.randint(0, 32)
        old_x, old_y = current_points[idx]
        
        # Random perturbation
        angle = random.uniform(0, 2*np.pi)
        dist = random.uniform(0, step_size)
        new_x = old_x + dist * math.cos(angle)
        new_y = old_y + dist * math.sin(angle)
        
        # Clamp to [0, 1]
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        
        current_points[idx] = [new_x, new_y]
        
        # Check validity
        if not is_valid(current_points):
            current_points[idx] = [old_x, old_y]
            continue
        
        new_count = count_convex_7goms(current_points)
        
        if new_count < current_count:
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_points = current_points.copy()
            no_improve_count = 0
            # Occasionally increase step size
            if random.random() < 0.1:
                step_size = min(0.1, step_size * 1.1)
        else:
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset to best and try different step size
                current_points = best_points.copy()
                current_count = best_count
                step_size = random.uniform(0.01, 0.1)
                no_improve_count = 0
    
    return best_points

# EVOLVE-BLOCK-END
