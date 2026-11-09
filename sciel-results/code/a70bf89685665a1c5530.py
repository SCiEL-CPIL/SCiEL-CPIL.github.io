import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
# i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside(points):
    """
    Check if all points are inside the equilateral triangle (0,0), (1,0), (0.5, H).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Check y >= 0
    if np.any(y < -1e-9):
        return False
    
    # Check y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
        
    # Check y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective(params):
    """
    Objective function for minimization.
    Returns -min_area if valid, penalty if invalid.
    """
    points = params.reshape(N, 2)
    
    if not is_inside(points):
        return 1.0  # Penalty: larger than any possible area (max area ~ 0.04)
        
    area = min_triangle_area_vec(points)
    return -area

def refine_local(points, start_time, deadline):
    """
    Perform local refinement using Nelder-Mead on the best point set.
    Returns the refined point set.
    """
    if time.time() > deadline:
        return points
    
    best_points = points.copy()
    best_score = min_triangle_area_vec(best_points)
    
    # Try a few local optimizations with different initial perturbations
    # This helps escape shallow local minima found by DE
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        # Create a slightly perturbed version to start from
        if attempt == 0:
            x0 = best_points.flatten()
        else:
            # Small random perturbation
            perturbation = rng.normal(0, 0.01, size=(N, 2))
            x0 = (best_points + perturbation).flatten()
            
            # Ensure validity by checking bounds roughly
            # If invalid, skip this perturbation
            if not is_inside(x0.reshape(N, 2)):
                continue
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if time.time() > deadline:
                break
                
            new_points = res.x.reshape(N, 2)
            if is_inside(new_points):
                new_score = min_triangle_area_vec(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
        except Exception:
            continue
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Bounds for Cartesian coordinates
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search
    try:
        # maxiter and popsize tuned for time budget
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,  # Reduced to leave time for local refinement
            popsize=15,
            tol=0.01,
            seed=42,
            polish=True,  # Use Nelder-Mead to polish the best solution
            init='sobol' # Better coverage of the search space
        )
        
        best_params = result.x
        candidate_points = best_params.reshape(N, 2)
        
        if is_inside(candidate_points):
            candidate_score = min_triangle_area_vec(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
    except Exception:
        pass
        
    # If DE didn't find a valid point or failed, try a few random restarts
    if best_points is None:
        # Fallback to a simple valid configuration
        rng = np.random.default_rng(42)
        for _ in range(100):
            # Sample in bounding box
            pts = rng.random((N, 2))
            pts[:, 0] *= 1.0
            pts[:, 1] *= H
            if is_inside(pts):
                best_points = pts
                best_score = min_triangle_area_vec(best_points)
                break
        if best_points is None:
            # Last resort: grid points
            best_points = np.array([[0.1*i, 0.1*j] for i in range(3) for j in range(3)][:N])
            # Ensure they are inside
            valid_pts = []
            for p in best_points:
                if is_inside(p.reshape(1, 2)):
                    valid_pts.append(p)
            if len(valid_pts) < N:
                # Generate more
                while len(valid_pts) < N:
                    p = rng.random(2)
                    p[0] *= 1.0
                    p[1] *= H
                    if is_inside(p.reshape(1, 2)):
                        valid_pts.append(p)
            best_points = np.array(valid_pts[:N])
            best_score = min_triangle_area_vec(best_points)
    
    # Check time before local refinement
    if time.time() < deadline - 10:
        # Perform local refinement to squeeze out more performance
        best_points = refine_local(best_points, start_time, deadline)
    
    return best_points
