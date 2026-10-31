import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(points):
    """
    Objective to maximize: min_area - penalty
    """
    area = _min_area(points)
    pen = _penalty(points)
    return area - 1000.0 * pen

def _smooth_objective(points, beta):
    """
    Smooth approximation of min area using log-sum-exp.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    # min(x) ≈ -logsumexp(-x)
    # We subtract a small offset to avoid underflow/overflow issues if areas are large
    # but areas are small here.
    min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    pen = _penalty(points)
    return min_approx - 1000.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _hexagonal_init(rng):
    """
    Generate an initial configuration based on a hexagonal lattice,
    slightly perturbed. This is a strong prior for packing problems.
    """
    points = np.zeros((N, 2))
    
    # Define a grid of points inside the triangle
    # We'll use a simple mapping from a square grid to the triangle
    # or just place them in a roughly hexagonal pattern.
    
    # A known good structure for 11 points often looks like:
    # Row 1: 1 point
    # Row 2: 2 points
    # Row 3: 3 points
    # Row 4: 3 points
    # Row 5: 2 points
    # (This is a rough guess, let's use a more systematic approach)
    
    # Let's use a set of pre-defined relative positions that form a good packing
    # and then scale/translate them to fit the triangle.
    
    # Alternatively, use a simple grid and clip/project.
    # Let's try a 3x4 grid clipped to the triangle, then add noise.
    
    xs = np.linspace(0.1, 0.9, 4)
    ys = np.linspace(0.1, 0.8, 3)
    
    grid_pts = []
    for x in xs:
        for y in ys:
            # Check if inside triangle
            if y <= S3 * x and y <= S3 * (1 - x) and y >= 0:
                grid_pts.append([x, y])
    
    # If we have more than 11, pick 11 randomly. If less, add randoms.
    if len(grid_pts) >= N:
        idxs = rng.choice(len(grid_pts), N, replace=False)
        points = np.array([grid_pts[i] for i in idxs])
    else:
        points = np.array(grid_pts)
        while points.shape[0] < N:
            p = _random_valid_point(rng)
            points = np.vstack([points, p])
    
    # Add small noise to break symmetry
    points += rng.normal(0, 0.01, points.shape)
    
    return points

def _local_refine(points, time_limit, rng):
    """
    Refine a configuration using L-BFGS-B on the smooth objective.
    """
    start_time = time.time()
    
    # Use a moderately high beta for smoothness
    beta = 100.0
    
    def _obj(flat_pts):
        pts = flat_pts.reshape(-1, 2)
        return -_smooth_objective(pts, beta)
    
    try:
        res = minimize(
            _obj, 
            points.flatten(), 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        refined_pts = res.x.reshape(-1, 2)
        return refined_pts
    except Exception:
        return points

def _simulated_annealing(initial_points, time_limit, rng):
    """
    Run Simulated Annealing.
    """
    current_points = initial_points.copy()
    current_score = _objective(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05
    cooling_rate = 0.999
    min_T = 1e-5
    
    step_scale = 0.05
    
    start_time = time.time()
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        idx = rng.integers(N)
        noise = rng.normal(0, step_scale, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        new_score = _objective(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 50:
            step_scale = max(0.001, step_scale * 0.9)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.1, step_scale * 1.05)
        
        T *= cooling_rate
        
        # Occasional large jump
        if rng.random() < 0.01:
             idx = rng.integers(N)
             new_points = current_points.copy()
             new_points[idx] = _random_valid_point(rng)
             new_score = _objective(new_points)
             delta = new_score - current_score
             if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin: evaluator kills at 150s
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start strategy
    # 1. Hexagonal init
    # 2. Random init
    # 3. Perturbed hexagonal init
    
    num_starts = 8
    time_per_start = (timeout - 30) / num_starts  # Leave 30s for final polishing
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 30:
            break
        
        if i == 0:
            init_pts = _hexagonal_init(rng)
        elif i == 1:
            init_pts = _hexagonal_init(rng)
            init_pts += rng.normal(0, 0.1, init_pts.shape)
        else:
            # Random init
            init_pts = np.array([_random_valid_point(rng) for _ in range(N)])
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Polish the best result found using L-BFGS-B with smooth objective
    if best_points is not None and time.time() - start_time < timeout - 10:
        # Try a few different beta values for polishing
        for beta in [50, 100, 200]:
            if time.time() - start_time > timeout - 5:
                break
            
            def _min_smooth_obj(flat_pts, beta_val=beta):
                pts = flat_pts.reshape(-1, 2)
                return -_smooth_objective(pts, beta_val)
            
            try:
                res = minimize(
                    _min_smooth_obj, 
                    best_points.flatten(), 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                polished_pts = res.x.reshape(-1, 2)
                polished_score = _objective(polished_pts)
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts
            except Exception:
                pass
    
    # Final validity check: clip points to triangle if slightly outside
    if best_points is None:
        best_points = _hexagonal_init(rng)
    
    # Simple clipping to ensure points are inside triangle
    x = best_points[:, 0]
    y = best_points[:, 1]
    
    # Clip y >= 0
    y = np.maximum(y, 0)
    
    # Clip y <= sqrt(3) * x
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    # Clip y <= sqrt(3) * (1-x)
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    # Re-check after clipping
    y = np.maximum(y, 0)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    best_points[:, 0] = x
    best_points[:, 1] = y
    
    return best_points
