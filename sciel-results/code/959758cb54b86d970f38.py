"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Exact Combinatorial Local Search (Hill Climbing).
Instead of smooth approximations, we directly manipulate points.
We identify the triangle with the minimum area (the 'critical' triangle).
We then attempt to move one of its vertices to increase this area, 
while ensuring no other triangle becomes smaller than the current target.
This is a form of active set method / coordinate descent on the combinatorial structure.
"""
import numpy as np
import random
import time
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

def _area(p1, p2, p3):
    """Calculate area of triangle formed by p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area_info(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    Returns: (min_area, (i, j, k)) where i,j,k are indices of the critical triangle.
    """
    n = len(points)
    min_area = float('inf')
    critical_idx = (0, 1, 2)
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
                    critical_idx = (i, j, k)
                    
    return (min_area / AREA, critical_idx)

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9

def _clip_to_triangle(p):
    """Clip a point to be inside the triangle."""
    if not _is_inside_triangle(p):
        # Simple projection logic: 
        # If outside, pull towards centroid or clip to edges.
        # For robustness in local search, we just revert if invalid.
        # But a better approach is to generate valid moves.
        pass
    return p

def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return np.array(pts)

def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)

def _local_search(points, rng, max_iters=1000, deadline=None):
    """
    Perform local search by perturbing points involved in critical triangles.
    """
    current_points = points.copy()
    current_min, critical_idx = _min_triangle_area_info(current_points)
    
    # Identify all triangles that are "near critical" (within 5% of min)
    # This helps in understanding constraints
    near_critical = set()
    n = len(current_points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _area(current_points[i], current_points[j], current_points[k]) / AREA
                if a < current_min * 1.05:
                    near_critical.add((i, j, k))
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        # Try moving each point in the critical triangle
        # And potentially points in near-critical triangles
        candidates = set(critical_idx)
        for idx in near_critical:
            for p in idx:
                candidates.add(p)
        
        for idx in candidates:
            p_orig = current_points[idx].copy()
            
            # Strategy 1: Move away from the opposite side of the critical triangle
            if idx in critical_idx:
                i, j, k = critical_idx
                # Find the other two points in the critical triangle
                others = [x for x in (i, j, k) if x != idx]
                p_a = current_points[others[0]]
                p_b = current_points[others[1]]
                
                # Vector from p_orig to the line segment p_a-p_b
                # We want to move p_orig away from this segment to increase area
                # Direction: normal to the segment, pointing away from the segment
                d = p_b - p_a
                # Normal vector
                n_vec = np.array([-d[1], d[0]])
                # Determine which side p_orig is on
                # Cross product (p_orig - p_a) x d
                cross = (p_orig[0] - p_a[0]) * d[1] - (p_orig[1] - p_a[1]) * d[0]
                if cross < 0:
                    n_vec = -n_vec
                # Normalize
                n_norm = np.linalg.norm(n_vec)
                if n_norm > 1e-9:
                    n_vec /= n_norm
                else:
                    continue
                
                # Try moving in this direction with varying step sizes
                for step in [0.01, 0.005, 0.02, 0.001]:
                    new_p = p_orig + n_vec * step
                    if _is_inside_triangle(new_p):
                        current_points[idx] = new_p
                        new_min, new_critical = _min_triangle_area_info(current_points)
                        if new_min > current_min + 1e-10:
                            current_min = new_min
                            critical_idx = new_critical
                            improved = True
                            # Update near critical set if needed, but for speed we can just re-evaluate
                            # For simplicity, we break out of step loop
                            break
                        else:
                            current_points[idx] = p_orig
            
            # Strategy 2: Random perturbation of the point
            # This helps escape local minima where "moving away" doesn't help due to other constraints
            for _ in range(5):
                noise = np.array([rng.gauss(0, 0.01), rng.gauss(0, 0.01)])
                new_p = p_orig + noise
                if _is_inside_triangle(new_p):
                    current_points[idx] = new_p
                    new_min, new_critical = _min_triangle_area_info(current_points)
                    if new_min > current_min + 1e-10:
                        current_min = new_min
                        critical_idx = new_critical
                        improved = True
                        break
                    else:
                        current_points[idx] = p_orig
                        
            # If we improved, we might want to restart the candidate loop or just continue
            # Restarting is safer to re-identify critical triangles
            if improved:
                # Re-identify near critical
                near_critical = set()
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            a = _area(current_points[i], current_points[j], current_points[k]) / AREA
                            if a < current_min * 1.05:
                                near_critical.add((i, j, k))
                break # Break inner for loop, restart while loop condition check
                
    return current_points, current_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # 1. Structured Start
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # 2. Random Starts
    # We can afford more random starts with this faster local search
    for _ in range(20):
        initial_configs.append(_generate_random_points(rng))
    
    # 3. Run Local Search on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # We can run multiple iterations of local search with different random seeds/perturbations
        # But the local_search function already does internal iteration.
        # Let's add a "shake" before local search to diversify
        shaken_pts = init_pts.copy()
        # Small random perturbation to escape symmetric traps
        for i in range(N):
            noise = np.array([rng.gauss(0, 0.005), rng.gauss(0, 0.005)])
            new_p = shaken_pts[i] + noise
            if _is_inside_triangle(new_p):
                shaken_pts[i] = new_p
        
        res_pts, res_min = _local_search(shaken_pts, rng, max_iters=200, deadline=deadline)
        
        if res_min > best_min:
            best_min = res_min
            best_points = res_pts.copy()
            
            # If we have time, we can try to polish the best solution further
            # by running local search again on the best solution with smaller steps
            if time.time() < deadline - 10:
                # Re-run local search on the best found so far
                polished_pts, polished_min = _local_search(best_points, rng, max_iters=500, deadline=deadline)
                if polished_min > best_min:
                    best_min = polished_min
                    best_points = polished_pts.copy()

    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
