"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Direct Search using Vectorized Nelder-Mead (Derivative-Free).
   - Avoids gradient calculations and explicit constraint handling overhead.
   - Robust to the non-smooth nature of the 'minimum area' objective.
2. High-Fidelity Multi-Start:
   - Uses a Greedy Max-Min Distance initialization to create diverse, 
     high-quality starting configurations.
   - Adds random perturbations to these bases for further diversity.
3. Vectorized Area Calculation:
   - Pre-computes triplet indices.
   - Uses NumPy broadcasting to calculate all 165 triangle areas in one shot.
   - This makes each Nelder-Mead function evaluation extremely fast, 
     allowing for more iterations and restarts within the time limit.
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
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # In 2D, cross product is scalar: x1*y2 - y1*x2
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Lines:
    # Bottom: y >= 0
    # Left: y <= sqrt(3) * x
    # Right: y <= sqrt(3) * (1 - x)
    
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
    
    # Clip y
    y = np.clip(y, 0, np.inf)
    
    # Clip based on left and right boundaries
    # y <= sqrt(3) * x  => x >= y / sqrt(3)
    # y <= sqrt(3) * (1 - x) => x <= 1 - y / sqrt(3)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # If x_min > x_max, the point is above the triangle apex for this y, 
    # or y is too high. 
    # Max y is S3H.
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # Clip x
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _objective(x_flat):
    """
    Objective function for minimization.
    We want to MAXIMIZE min_area, so we MINIMIZE -min_area.
    x_flat: shape (N*2,)
    """
    pts = x_flat.reshape(N, 2)
    
    # Projection is a hard constraint enforcement.
    # If we use Nelder-Mead, it might step outside.
    # Projecting back ensures validity but creates non-smoothness.
    # However, for this problem, staying inside is critical.
    pts = _project_to_triangle(pts)
    
    area = _min_triangle_area(pts)
    return -area

def _greedy_init(rng):
    """
    Greedy initialization: place points one by one to maximize the minimum distance to existing points.
    This is a strong heuristic for packing.
    """
    pts = []
    
    # Start with a random point in the center region to avoid corners initially?
    # Or just a random point.
    # Let's pick a point.
    def _random_point_in_triangle(rng):
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        x = u + 0.5 * v
        y = S3H * v
        return [x, y]

    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        
        # Sample candidates
        # Adaptive sampling: more candidates if we are early?
        # 1000 candidates is fast enough.
        for _ in range(1000):
            cand = np.array(_random_point_in_triangle(rng))
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        
        if best_pos is not None:
            pts.append(best_pos)
            
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Multi-start Nelder-Mead with Greedy Initialization.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # 1. Generate a set of diverse initializations
    # We will use the Greedy method to generate base configurations,
    # then perturb them.
    
    initializations = []
    
    # Generate 5 distinct greedy bases
    num_bases = 5
    for _ in range(num_bases):
        try:
            base_pts = _greedy_init(rng)
            initializations.append(base_pts)
            
            # Create 2 perturbations of this base
            for _ in range(2):
                perturbed = base_pts + np.random.normal(0, 0.05, (N, 2))
                perturbed = _project_to_triangle(perturbed)
                initializations.append(perturbed)
        except Exception:
            pass
            
    # Add a few purely random ones just in case
    for _ in range(5):
        # Random points
        u = np.random.random((N, 2))
        mask = u[:, 0] + u[:, 1] > 1
        u[mask] = 1 - u[mask]
        x = u[:, 0] + 0.5 * u[:, 1]
        y = S3H * u[:, 1]
        rand_pts = np.column_stack((x, y))
        initializations.append(rand_pts)
        
    # Shuffle initializations to ensure variety in early runs
    rng.shuffle(initializations)
    
    # 2. Optimize each initialization
    # Nelder-Mead parameters
    # maxiter: max function evaluations per simplex
    # xatol, fatol: tolerances
    # We want to converge reasonably well but not spend too much time on one.
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000,  # Allow many iterations
                    'xatol': 1e-5, 
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            pts_opt = res.x.reshape(N, 2)
            pts_opt = _project_to_triangle(pts_opt)
            
            if _check_inside(pts_opt):
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            # If optimization fails, evaluate the initial point
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
            continue
            
        # Check time again
        if time.time() - start_time > DEADLINE - 10:
            break

    # 3. Fallback
    if best_pts is None:
        # Use the best from initializations without optimization
        for init_pts in initializations:
            if _check_inside(init_pts):
                score = _min_triangle_area(init_pts)
                if score > best_score:
                    best_score = score
                    best_pts = init_pts.copy()
                    
    if best_pts is None:
        # Absolute fallback: grid points
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
