"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from itertools import combinations

N = 11
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLES.shape[0]

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # y must be between 0 and H
    y = np.clip(y, 0, H)
    
    # x bounds depend on y
    # Left edge: x = y / sqrt(3)
    # Right edge: x = 1 - y / sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _min_area(points):
    """Compute the smallest triangle area among all triples."""
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective_smooth(points, power=2.0, k=5):
    """
    Compute a smoothed objective function based on the k smallest areas.
    Using a power mean with negative power emphasizes the small areas
    but is smoother than the strict minimum.
    """
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Get the k smallest areas
    # We want to maximize the minimum, so we minimize the negative power mean of the smallest areas.
    # Actually, let's just minimize the sum of the reciprocals of the smallest areas, or a power mean.
    # Let's use the k-th smallest area as a proxy, but smoothed.
    # A common trick: minimize sum(1/area) for the smallest few, or just use the geometric mean of the smallest k.
    # Let's try: - (geometric mean of the smallest k areas)
    
    areas_sorted = np.sort(areas)
    smallest_k = areas_sorted[:k]
    
    # Avoid division by zero
    smallest_k = np.maximum(smallest_k, 1e-10)
    
    # Geometric mean
    log_sum = np.sum(np.log(smallest_k)) / k
    geo_mean = np.exp(log_sum)
    
    # We want to maximize this, so return negative
    return -geo_mean

def _simulated_annealing(points0, t0=1.0, t_end=1e-6, cooling=0.995, max_iter=5000, step_size=0.1, deadline=None):
    """
    Simulated Annealing to maximize the minimum triangle area.
    Uses a smoothed objective to navigate the landscape.
    """
    current_points = points0.copy()
    current_obj = _objective_smooth(current_points)
    best_points = current_points.copy()
    best_obj = current_obj
    best_min_area = _min_area(current_points)
    
    rng = np.random.RandomState(42)
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        temp = t0 * (cooling ** i)
        if temp < t_end:
            break
            
        # Generate a new candidate by perturbing one point
        new_points = current_points.copy()
        idx = rng.randint(N)
        noise = rng.normal(0, step_size, size=2)
        new_points[idx] += noise
        new_points = _clip_to_triangle(new_points)
        
        new_obj = _objective_smooth(new_points)
        delta_obj = new_obj - current_obj
        
        # Accept if better, or with probability exp(-delta/T)
        if delta_obj > 0 or (temp > 0 and rng.rand() < np.exp(-delta_obj / temp)):
            current_points = new_points
            current_obj = new_obj
            
            # Check if we improved the actual minimum area (the true objective)
            current_min = _min_area(current_points)
            if current_min > best_min_area:
                best_min_area = current_min
                best_points = current_points.copy()
        
        # Adaptive step size: if stuck, increase; if improving, decrease slightly
        # Simple heuristic: if no improvement in last 100 steps, increase step
        # For simplicity, we'll just use a fixed schedule or a simple decay
        
    return best_points, best_min_area

def _local_refine_nm(points, maxiter=200):
    """
    Refine using Nelder-Mead on the smoothed objective, then verify with min area.
    """
    def obj(x):
        pts = x.reshape(-1, 2)
        pts = _clip_to_triangle(pts)
        return _objective_smooth(pts, power=1.0, k=3)
        
    try:
        from scipy.optimize import minimize
        res = minimize(obj, points.flatten(), method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        new_points = _clip_to_triangle(res.x.reshape(-1, 2))
        return new_points
    except:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin
    
    best_points = None
    best_score = -np.inf
    
    # 1. Structured Initialization: Triangular Grid
    # For n=11, a 4x4 grid has 16 points, we can pick 11.
    # Or a hexagonal packing.
    # Let's try a few structured starts.
    
    # Start 1: Corners and midpoints and center
    # (0,0), (1,0), (0.5, H), (0.5, 0), (0.25, H/2), (0.75, H/2), (0.5, H/2), etc.
    # A good heuristic is to place points on a grid in barycentric coordinates.
    
    inits = []
    
    # Grid based init
    # Barycentric coords (u, v) with u+v <= 1
    # Grid size 4x4
    grid_points = []
    for i in range(4):
        for j in range(4 - i):
            u = i / 3.0
            v = j / 3.0
            # Convert to Cartesian
            x = u + 0.5 * v
            y = v * (np.sqrt(3) / 2.0)
            grid_points.append([x, y])
            
    # We have 10 points in a 4x4 triangular grid (0..3). 
    # 1+2+3+4 = 10. We need 11.
    # Let's add the centroid or subdivide.
    # Actually, a 4x4 grid in barycentric indices 0,1,2,3 gives 10 points.
    # To get 11, we can use a 5x5 grid and pick the best 11, or just add one more.
    # Let's use a slightly finer grid or random perturbation of the grid.
    
    # Init 1: The 10 grid points + centroid
    centroid = [0.5, H/3] # Centroid of equilateral triangle
    init1 = np.array(grid_points + [centroid])
    inits.append(init1)
    
    # Init 2: Random
    rng = np.random.RandomState(123)
    for _ in range(5):
        pts = rng.uniform(0, 1, (N, 2))
        pts = _clip_to_triangle(pts)
        inits.append(pts)
        
    # Init 3: Perturbed Grid
    for scale in [0.05, 0.1, 0.2]:
        pts = init1 + rng.normal(0, scale, size=init1.shape)
        pts = _clip_to_triangle(pts)
        inits.append(pts)
        
    # Run Simulated Annealing on each init
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Run SA with different parameters
        # Longer run for the first few, shorter for others
        max_iter = 3000 if i < 3 else 1000
        
        sa_pts, sa_score = _simulated_annealing(
            init_pts, 
            t0=0.5, 
            cooling=0.998, 
            max_iter=max_iter, 
            step_size=0.15,
            deadline=deadline - 5
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Refine the best SA result with Nelder-Mead
        if time.time() > deadline - 20:
            break
        if best_points is not None:
            refined_pts = _local_refine_nm(best_points, maxiter=100)
            refined_score = _min_area(refined_pts)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
                
    # Final polish: Run SA again on the best point found, with smaller steps
    if best_points is not None and time.time() < deadline - 5:
        final_pts, final_score = _simulated_annealing(
            best_points, 
            t0=0.1, 
            cooling=0.999, 
            max_iter=2000, 
            step_size=0.05,
            deadline=deadline - 2
        )
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
            
        # One last NM polish
        if time.time() < deadline - 1:
            refined_pts = _local_refine_nm(best_points, maxiter=50)
            refined_score = _min_area(refined_pts)
            if refined_score > best_score:
                best_points = refined_pts.copy()
                
    if best_points is None:
        # Fallback
        best_points = _clip_to_triangle(np.array(grid_points + [centroid]))
        
    return best_points
