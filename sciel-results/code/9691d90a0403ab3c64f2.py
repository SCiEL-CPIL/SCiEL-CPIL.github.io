# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute triplets for vectorized area calculation
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _min_triangle_area(pts):
    """Calculate the minimum area of any triangle formed by the points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _project_to_triangle(pts):
    """Project points onto the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Clip y to [0, S3H]
        y = np.clip(y, 0.0, S3H)
        
        # Determine x bounds based on y
        # Left edge: x = y / S3
        # Right edge: x = 1 - y / S3
        x_min = y / S3
        x_max = 1.0 - y / S3
        
        # If y is very close to S3H, x_min and x_max converge to 0.5
        x = np.clip(x, x_min, x_max)
        pts[i] = [x, y]
    return pts

def construct():
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialization: Use a mix of grid and random points to ensure diversity
    # A structured initialization often helps SA start in a decent region
    # We will use a simple random initialization but ensure points are inside
    pts = np.random.rand(N, 2)
    pts[:, 0] *= 1.0
    pts[:, 1] *= S3H
    # Simple rejection sampling to get points inside triangle
    for i in range(N):
        while True:
            x = rng.random()
            y = rng.random() * S3H
            if y <= S3H * (1 - x) and y <= S3H * x: 
                 # Wait, triangle is (0,0), (1,0), (0.5, S3H)
                 # Left bound: x >= y/S3 => y <= S3*x
                 # Right bound: x <= 1 - y/S3 => y <= S3*(1-x)
                 # So condition is y <= min(S3*x, S3*(1-x))
                 if y <= min(S3 * x, S3 * (1 - x)):
                     pts[i] = [x, y]
                     break
    pts = _project_to_triangle(pts)
    
    current_score = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.01 # Initial temperature
    T_min = 1e-6
    alpha = 0.99 # Cooling rate
    max_steps = 100000 # Max steps to run
    
    step_size = 0.05
    
    for step in range(max_steps):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate new candidate
        new_pts = best_pts.copy()
        # Pick a random subset of points to perturb (or all, but smaller step)
        # Perturbing all points with small steps is safer for constraint handling
        new_pts += rng.normal(0, step_size, size=(N, 2))
        new_pts = _project_to_triangle(new_pts)
        
        new_score = _min_triangle_area(new_pts)
        
        # Acceptance criterion
        delta = new_score - best_score
        if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
            best_pts = new_pts
            best_score = new_score
            current_score = new_score
        else:
            # If rejected, we might want to keep the current best and just cool
            pass
            
        # Cool down
        T *= alpha
        if T < T_min:
            # Restart with higher temperature to escape local minima
            T = 0.01
            step_size *= 0.95 # Reduce step size slightly over restarts? Or keep constant.
            # Let's reduce step size to refine
            if step_size < 0.001:
                break

    # Final Polish: A few iterations of local search on the best point found
    # Using a simple coordinate descent or small random steps
    if time.time() - start_time < DEADLINE - 5:
        polish_steps = 2000
        for i in range(polish_steps):
            if time.time() - start_time > DEADLINE - 2:
                break
            new_pts = best_pts + rng.normal(0, 0.005, size=(N, 2))
            new_pts = _project_to_triangle(new_pts)
            new_score = _min_triangle_area(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score

    return best_pts
# EVOLVE-BLOCK-END
