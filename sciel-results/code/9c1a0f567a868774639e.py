"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Structured Lattice Representation:
   - Partition points into concentric layers (rings) based on radial distance from centroid.
   - Layer 0: 1 point (center)
   - Layer 1: 3 points (triangle)
   - Layer 2: 7 points (heptagon)
   - This structure inherently promotes separation and symmetry.
2. Parameterization:
   - Variables: radial scale for each layer, angular rotation for each layer, 
     and small individual angular perturbations for points in larger layers to break symmetry if beneficial.
   - This reduces the search space from 22D raw coordinates to a lower-dimensional 
     structured space (~10-12 dims), making local search much more effective.
3. Optimization:
   - Use a fast, vectorized coordinate-descent / Nelder-Mead hybrid on the structural parameters.
   - The objective remains the raw minimum triangle area.
   - Multi-start with different initial radial scales and rotations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    if np.any(y < -eps): return False
    if np.any(y > np.sqrt(3) * x + eps): return False
    if np.any(y > np.sqrt(3) * (1 - x) + eps): return False
    if np.any(x < -eps): return False
    if np.any(x > 1 + eps): return False
    
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _params_to_points(params):
    """
    Convert structured parameters to Cartesian coordinates.
    params structure:
    - r0: radius for layer 0 (center, effectively 0)
    - r1: radius for layer 1 (3 points)
    - r2: radius for layer 2 (7 points)
    - theta1: rotation for layer 1
    - theta2: rotation for layer 2
    - perturbations: small angular deviations for points in layer 2 (7 values)
    - perturbations: small angular deviations for points in layer 1 (3 values)
    
    Total dims: 3 (radii) + 2 (rotations) + 10 (perturbations) = 15 dims.
    We can optimize these.
    """
    r0 = params[0]
    r1 = params[1]
    r2 = params[2]
    theta1 = params[3]
    theta2 = params[4]
    
    # Layer 0: 1 point at center (0,0) in local coords, but we map to triangle centroid later?
    # Actually, it's easier to work in a coordinate system centered at the triangle's centroid.
    # Centroid of (0,0), (1,0), (0.5, S3H) is (0.5, S3H/3).
    cx, cy = 0.5, S3H / 3.0
    
    pts = []
    
    # Layer 0
    pts.append([cx + r0, cy]) # r0 should be near 0
    
    # Layer 1: 3 points
    for i in range(3):
        angle = theta1 + i * 2 * np.pi / 3
        # Add small perturbation if we had them, but for simplicity, just rotation
        # Let's add perturbations from params[5:8]
        p_angle = angle + params[5 + i]
        x = cx + r1 * np.cos(p_angle)
        y = cy + r1 * np.sin(p_angle)
        pts.append([x, y])
        
    # Layer 2: 7 points
    for i in range(7):
        angle = theta2 + i * 2 * np.pi / 7
        # Perturbations from params[8:15]
        p_angle = angle + params[8 + i]
        x = cx + r2 * np.cos(p_angle)
        y = cy + r2 * np.sin(p_angle)
        pts.append([x, y])
        
    return np.array(pts)

def _objective_structured(params):
    """
    Objective for structured parameter optimization.
    """
    pts = _params_to_points(params)
    # Ensure points are inside. If not, penalize heavily or project.
    # Projecting is safer for continuity in search, but might hide bad params.
    # Let's project to ensure validity, then calculate area.
    pts = _project_to_triangle(pts)
    
    if not _check_inside(pts):
        return 1e6 # Penalty
    
    area = _min_triangle_area(pts)
    return -area

def _generate_initial_params(rng):
    """
    Generate initial parameters for the structured lattice.
    Radii should be scaled to fit within the triangle.
    Max radius from centroid to vertex is approx 0.5 / sin(60) * something?
    Distance from centroid (0.5, S3H/3) to vertex (0,0):
    dx = 0.5, dy = S3H/3.
    dist = sqrt(0.25 + 3/36) = sqrt(0.25 + 1/12) = sqrt(3/12 + 1/12) = sqrt(4/12) = sqrt(1/3) = 1/sqrt(3) ≈ 0.577.
    
    So r_max ≈ 0.577.
    r2 should be close to r_max to push points to boundaries.
    r1 should be roughly r2 * 0.5.
    r0 = 0.
    """
    r_max = 0.577
    r2 = r_max * rng.uniform(0.8, 1.0)
    r1 = r2 * rng.uniform(0.4, 0.6)
    r0 = 0.0
    
    theta1 = rng.uniform(0, 2*np.pi)
    theta2 = rng.uniform(0, 2*np.pi)
    
    # Small random perturbations
    perts1 = rng.uniform(-0.1, 0.1, 3)
    perts2 = rng.uniform(-0.1, 0.1, 7)
    
    params = np.array([r0, r1, r2, theta1, theta2] + list(perts1) + list(perts2))
    return params

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Structured Lattice Optimization.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start optimization
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        params0 = _generate_initial_params(rng)
        
        try:
            # Use Nelder-Mead on the structured parameters
            # It's robust and handles the small non-smoothness from projection well
            res = minimize(
                _objective_structured,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            params_opt = res.x
            pts_opt = _params_to_points(params_opt)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception:
            continue
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break

    # If structured search failed or didn't find anything, fallback to simple greedy
    if best_pts is None:
        # Simple fallback: Grid-like points
        best_pts = np.array([
            [0.5, S3H/3], # Center
            [0.25, S3H/6], [0.75, S3H/6], [0.5, 0], # Inner triangle
            [0.1, S3H/10], [0.9, S3H/10], [0.5, S3H/2], # Outer ring approx
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H], # Vertices
            [0.25, 0.0], [0.75, 0.0] # Midpoints
        ])
        best_pts = _project_to_triangle(best_pts)
        
    # Final safety check and projection
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
