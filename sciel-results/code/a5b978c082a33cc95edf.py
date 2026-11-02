"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Symmetry-Constrained Low-Dimensional Optimization.
Instead of optimizing 22 coordinates independently, we model the 11 points
using a structured triangular lattice with a few global parameters:
1. Global scale factor
2. Global rotation
3. Radial scaling factor for the outer ring (if applicable)
4. Small symmetry-breaking perturbations (optional)

This reduces the search space significantly, allowing the optimizer to
navigate the landscape more effectively. We use Nelder-Mead or Powell
on these few parameters, and then perform a final local polish on the
individual points if time permits.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _generate_lattice_points(scale=1.0, rotation=0.0, center=(0.5, S3H/3)):
    """
    Generate 11 points based on a triangular lattice structure.
    Structure:
    - 1 center point
    - 6 points in a hexagonal ring around the center (radius r1)
    - 4 points in an outer ring or corners? 
    Actually, for 11 points, a common configuration is:
    - 1 center
    - 6 neighbors (hexagon)
    - 4 more? 
    Let's try a 4x4 triangular grid subset? 
    Or better: Use the known "best" configuration for 11 points which is often symmetric.
    
    Let's define a base set of relative coordinates in a unit triangle centered at origin,
    then scale, rotate, and translate.
    
    Base configuration (relative to centroid of the big triangle):
    The big triangle vertices are (0,0), (1,0), (0.5, S3H).
    Centroid is (0.5, S3H/3).
    
    Let's define points in a local coordinate system where the centroid is (0,0).
    We want 11 points.
    
    A good symmetric starting point:
    1. Center: (0,0)
    2. Hexagon at radius r1: 6 points at angles 0, 60, ..., 300
    3. Outer ring at radius r2: 6 points? That would be 13.
    4. Maybe 4 points in a square?
    
    Let's try:
    - 1 center
    - 6 points forming a hexagon
    - 4 points forming a larger square or diamond?
    
    Alternatively, use a triangular lattice:
    Layer 0: 1 point (center)
    Layer 1: 6 points (hexagon)
    Layer 2: 12 points (hexagon) -> too many.
    
    Let's stick to:
    1 center
    6 inner ring
    4 outer points placed symmetrically?
    
    Actually, the optimal 11-point configuration is often close to a 4x4 grid in barycentric coordinates
    or a specific symmetric pattern.
    
    Let's use a parameterized family:
    Points are defined by:
    - Center (0,0)
    - 6 points at radius r1, angles theta_i + rotation
    - 4 points at radius r2, angles phi_j + rotation
    
    We optimize r1, r2, rotation, and possibly a global scale/translation.
    But translation is fixed to the centroid for symmetry.
    
    Let's try:
    p0 = center
    p1..p6 = hexagon at r1
    p7..p10 = 4 points at r2, angles 45, 135, 225, 315 (square)
    
    This is 11 points.
    """
    cx, cy = center
    pts = []
    # Center
    pts.append([cx, cy])
    
    # Inner Hexagon
    for i in range(6):
        angle = i * np.pi / 3 + rotation
        x = cx + r1 * np.cos(angle)
        y = cy + r1 * np.sin(angle)
        pts.append([x, y])
        
    # Outer Square (4 points)
    # To maintain some symmetry with the triangle, maybe align the square with the triangle axes?
    # Or just a square.
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4 + rotation # 45, 135, 225, 315
        x = cx + r2 * np.cos(angle)
        y = cy + r2 * np.sin(angle)
        pts.append([x, y])
        
    return np.array(pts)

def _symmetry_objective(params):
    """
    Objective function for the symmetry-constrained optimization.
    params: [r1, r2, rotation, scale]
    We also need to ensure points are inside the triangle.
    If not, we apply a heavy penalty.
    """
    r1, r2, rotation, scale = params
    
    # Scale the radii
    r1_eff = r1 * scale
    r2_eff = r2 * scale
    
    # Center is fixed at the centroid of the big triangle
    center = (0.5, S3H / 3)
    
    # Generate points
    pts = _generate_lattice_points(r1_eff, r2_eff, rotation, center)
    
    # Check if points are inside
    penalty = 0.0
    for p in pts:
        x, y = p
        d1 = y
        d2 = np.sqrt(3) * x - y
        d3 = np.sqrt(3) * (1 - x) - y
        
        if d1 < 0: penalty += d1**2
        if d2 < 0: penalty += d2**2
        if d3 < 0: penalty += d3**2
        
        # Also check x bounds
        if x < 0: penalty += x**2
        if x > 1: penalty += (x-1)**2
        if y < 0: penalty += y**2
        if y > S3H: penalty += (y-S3H)**2
        
    if penalty > 1e-6:
        return 1000.0 * penalty
    
    # Calculate min area
    area = _min_triangle_area(pts)
    return -area

def _local_refine(pts):
    """
    Perform a local refinement on the individual points using coordinate descent.
    This helps to break the symmetry if it's slightly suboptimal.
    """
    current_pts = pts.copy()
    best_area = _min_triangle_area(current_pts)
    
    # Use a simple coordinate descent with a few iterations
    for _ in range(10): # Limit iterations for speed
        for idx in range(N):
            # Try moving point idx in small steps
            best_x = current_pts[idx].copy()
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    new_pt = current_pts[idx] + [dx, dy]
                    # Project to triangle
                    x, y = new_pt
                    if y < 0: y = 0
                    max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
                    if y > max_y: y = max_y
                    if x < 0: x = 0
                    if x > 1: x = 1
                    new_pt = [x, y]
                    
                    current_pts[idx] = new_pt
                    if _check_inside(current_pts):
                        area = _min_triangle_area(current_pts)
                        if area > best_area:
                            best_area = area
                            best_x = new_pt
                        else:
                            current_pts[idx] = best_x
                    else:
                        current_pts[idx] = best_x
                        
    return current_pts

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses symmetry-constrained optimization followed by local refinement.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # Initial guesses for parameters [r1, r2, rotation, scale]
    # r1 ~ 0.1, r2 ~ 0.2, rotation ~ 0, scale ~ 1
    initial_params = [0.1, 0.2, 0.0, 1.0]
    
    # Try multiple initializations
    n_starts = 5
    for i in range(n_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Perturb initial parameters
        params = initial_params.copy()
        params[0] += rng.uniform(-0.05, 0.05)
        params[1] += rng.uniform(-0.05, 0.05)
        params[2] += rng.uniform(-0.5, 0.5)
        params[3] += rng.uniform(-0.2, 0.2)
        
        # Optimize parameters
        try:
            result = minimize(
                _symmetry_objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < 0: # Valid solution (negative objective means positive area)
                r1, r2, rotation, scale = result.x
                center = (0.5, S3H / 3)
                pts = _generate_lattice_points(r1 * scale, r2 * scale, rotation, center)
                
                if _check_inside(pts):
                    area = _min_triangle_area(pts)
                    if area > best_score:
                        best_score = area
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # If we found a good symmetric solution, refine it locally
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        refined_pts = _local_refine(best_pts)
        if _check_inside(refined_pts):
            area = _min_triangle_area(refined_pts)
            if area > best_score:
                best_score = area
                best_pts = refined_pts
    
    # If symmetry approach failed or found nothing, fallback to a simple grid search
    if best_pts is None or best_score < 0.01:
        # Simple grid search for 11 points? No, too expensive.
        # Fallback to random initialization + local refine
        pts = np.array([[_random_point_in_triangle(rng)] for _ in range(N)])
        pts = _project_to_triangle(pts)
        refined_pts = _local_refine(pts)
        if _check_inside(refined_pts):
            area = _min_triangle_area(refined_pts)
            if area > best_score:
                best_score = area
                best_pts = refined_pts

    if best_pts is None:
        # Absolute fallback
        best_pts = np.array([[_random_point_in_triangle(rng)] for _ in range(N)])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]
