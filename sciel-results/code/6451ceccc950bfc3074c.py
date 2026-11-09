# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615  # Safety margin before 630s limit
    
    n = 33
    C_29_3 = 3654  # C(29, 3)
    
    # Precompute all 4-subsets of indices for fast delta counting
    all_4_subsets = list(combinations(range(n), 4))
    # Map: for each point index, which 4-subsets contain it
    point_to_subsets = [[] for _ in range(n)]
    for idx, subset in enumerate(all_4_subsets):
        for p in subset:
            point_to_subsets[p].append(idx)
    
    def is_convex_quad(p0, p1, p2, p3):
        """
        Check if 4 points are in convex position.
        A set of 4 points is in convex position iff no point is inside the triangle of the other 3.
        """
        def area2(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        
        def point_in_triangle(pt, v1, v2, v3):
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
        Count convex 7-gons.
        C = sum_{S in C(n,4), S is convex} C(n-4, 3)
        """
        total_count = 0
        for subset in all_4_subsets:
            p0 = points[subset[0]]
            p1 = points[subset[1]]
            p2 = points[subset[2]]
            p3 = points[subset[3]]
            if is_convex_quad(p0, p1, p2, p3):
                total_count += C_29_3
        return total_count

    def count_convex_7goms_delta(points, moved_idx, old_points, convex_status):
        """
        Calculate change in count when moving point at moved_idx.
        Only 4-subsets containing moved_idx can change status.
        """
        old_p = old_points[moved_idx]
        new_p = points[moved_idx]
        
        delta = 0
        for subset_idx in point_to_subsets[moved_idx]:
            subset = all_4_subsets[subset_idx]
            # Get the other 3 points
            others = [s for s in subset if s != moved_idx]
            p1 = points[others[0]]
            p2 = points[others[1]]
            p3 = points[others[2]]
            
            # Check new convexity
            new_convex = is_convex_quad(new_p, p1, p2, p3)
            
            # Old convexity is stored in convex_status[subset_idx]
            old_convex = convex_status[subset_idx]
            
            if old_convex and not new_convex:
                delta -= C_29_3
            elif not old_convex and new_convex:
                delta += C_29_3
            # Update status for future use
            convex_status[subset_idx] = new_convex
            
        return delta

    def is_valid(points):
        if points.shape != (33, 2):
            return False
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            return False
        # Distances
        for i in range(33):
            for j in range(i+1, 33):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                if dx*dx + dy*dy < 1e-12:
                    return False
        # Collinearity
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

    # Strategy: Start with a "clustered" structure (e.g., 3 dense clusters) 
    # which naturally minimizes convex subsets, then refine with SA.
    # Clusters reduce the number of points on the convex hull and 
    # increase the chance that 4-point subsets are non-convex.
    
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with 3 clusters
    centers = [
        (0.2, 0.2),
        (0.8, 0.2),
        (0.5, 0.8)
    ]
    
    pts = np.zeros((33, 2))
    for i in range(33):
        cluster_idx = i % 3
        cx, cy = centers[cluster_idx]
        pts[i, 0] = cx + np.random.normal(0, 0.05)
        pts[i, 1] = cy + np.random.normal(0, 0.05)
    
    # Clip to valid range
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity (add jitter if needed)
    if not is_valid(pts):
        pts += np.random.uniform(-0.001, 0.001, size=(33, 2))
        pts = np.clip(pts, 0.01, 0.99)
    
    if not is_valid(pts):
        # Fallback to random
        pts = np.random.rand(33, 2) * 0.98 + 0.01

    # Initialize convex status
    convex_status = np.zeros(len(all_4_subsets), dtype=bool)
    for idx, subset in enumerate(all_4_subsets):
        p0 = pts[subset[0]]
        p1 = pts[subset[1]]
        p2 = pts[subset[2]]
        p3 = pts[subset[3]]
        convex_status[idx] = is_convex_quad(p0, p1, p2, p3)
    
    current_count = count_convex_7goms(pts)
    best_points = pts.copy()
    best_count = current_count
    best_convex_status = convex_status.copy()
    
    # Simulated Annealing
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.9995
    step_size = 0.05
    
    # Main SA Loop
    while time.time() < deadline:
        # Move a random point
        idx = random.randint(0, 32)
        old_p = best_points[idx].copy()
        
        # Generate new position
        new_p = old_p + np.random.normal(0, step_size, 2)
        new_p = np.clip(new_p, 0.0, 1.0)
        
        # Check validity locally (distance and collinearity with other points)
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
        for i in range(33):
            if i == idx: continue
            for j in range(i+1, 33):
                if j == idx: continue
                x1, y1 = best_points[i]
                x2, y2 = best_points[j]
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
        convex_status_copy = best_convex_status.copy()
        delta = count_convex_7goms_delta(points_copy, idx, best_points, convex_status_copy)
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            best_points[idx] = new_p
            current_count += delta
            best_convex_status = convex_status_copy
            if current_count < best_count:
                best_count = current_count
            # Cool down
            T = max(T_min, T * cooling_rate)
            # Occasionally increase step size?
            if random.random() < 0.01:
                step_size = min(0.1, step_size * 1.05)
        else:
            # Reject
            T = max(T_min, T * cooling_rate)
            
        # Safety check every 1000 iterations
        if time.time() > deadline - 10:
            break

    # Final validation
    if not is_valid(best_points):
        # Fallback
        return np.random.rand(33, 2) * 0.98 + 0.01
        
    return best_points

# EVOLVE-BLOCK-END
