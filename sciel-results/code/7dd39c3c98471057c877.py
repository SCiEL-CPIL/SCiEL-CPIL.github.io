# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def compute_distances(points):
    """Compute all pairwise distances using scipy."""
    return spd.pdist(points)

def smooth_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for (d_min / d_max)^2.
    Uses log-sum-exp for soft max and min.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    exp_d = np.exp((dists - max_d) / beta)
    soft_max = max_d + beta * np.log(np.sum(exp_d))
    
    exp_neg_d = np.exp((min_d - dists) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_neg_d))
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio * ratio

def complex_to_points(z):
    """Convert complex numbers to 2D points, with z[0] fixed at origin."""
    # z[0] is fixed at 0 (origin)
    # z[1:] are the other 15 points
    n = len(z)
    points = np.zeros((n, 2))
    for i in range(1, n):
        points[i, 0] = z[i].real
        points[i, 1] = z[i].imag
    return points

def points_to_complex(points):
    """Convert 2D points to complex numbers, with first point at origin."""
    n = len(points)
    z = np.zeros(n, dtype=complex)
    # Translate so first point is at origin
    offset = points[0]
    for i in range(1, n):
        z[i] = complex(points[i, 0] - offset[0], points[i, 1] - offset[1])
    return z

def objective_complex(z_flat, beta=0.1):
    """
    Objective function for complex representation.
    z_flat is a real array of length 2*(n-1) representing the real and imaginary parts of z[1:].
    z[0] is fixed at 0.
    """
    n = 16
    z = np.zeros(n, dtype=complex)
    z[1:] = z_flat[0::2] + 1j * z_flat[1::2]
    
    points = complex_to_points(z)
    return -smooth_min_max_ratio(points, beta)

def gradient_complex(z_flat, beta=0.1):
    """
    Numerical gradient for complex representation.
    """
    n = 16
    grad = np.zeros_like(z_flat)
    h = 1e-7
    for i in range(len(z_flat)):
        z_plus = z_flat.copy()
        z_plus[i] += h
        f_plus = objective_complex(z_plus, beta)
        
        z_minus = z_flat.copy()
        z_minus[i] -= h
        f_minus = objective_complex(z_minus, beta)
        
        grad[i] = (f_plus - f_minus) / (2 * h)
    
    return grad

def lbfgsb_optimize(points, time_limit, beta=0.1, seed=None):
    """
    Optimize using L-BFGS-B with complex number representation.
    """
    n = 16
    z = points_to_complex(points)
    z_flat = np.zeros(2 * (n - 1))
    z_flat[0::2] = z[1:].real
    z_flat[1::2] = z[1:].imag
    
    # Set bounds: each coordinate in [-2, 2] (scale is invariant, so this is fine)
    bounds = [(-2.0, 2.0)] * (2 * (n - 1))
    
    start_time = time.time()
    best_z_flat = z_flat.copy()
    best_score = calculate_score(points)
    
    # Run L-BFGS-B with time limit
    try:
        res = minimize(objective_complex, z_flat, 
                      method='L-BFGS-B',
                      jac=gradient_complex,
                      bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        
        if res.success or res.fun < objective_complex(best_z_flat, beta):
            trial_z_flat = res.x
            trial_points = complex_to_points(np.concatenate([[0], 
                                                              trial_z_flat[0::2] + 1j * trial_z_flat[1::2]]))
            trial_score = calculate_score(trial_points)
            if trial_score > best_score:
                best_score = trial_score
                best_z_flat = trial_z_flat
    except Exception:
        pass
    
    # Convert back to points
    z_final = np.zeros(n, dtype=complex)
    z_final[1:] = best_z_flat[0::2] + 1j * best_z_flat[1::2]
    best_points = complex_to_points(z_final)
    
    return best_points, best_score

def local_search(points, time_limit, seed=None, use_smooth=False, beta=0.1):
    """
    Perform local search to maximize the score.
    Uses simulated annealing with adaptive step size.
    """
    n = points.shape[0]
    current_points = points.copy()
    
    if use_smooth:
        current_score = smooth_min_max_ratio(current_points, beta)
    else:
        current_score = calculate_score(current_points)
        
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    temp = 0.1
    min_temp = 1e-6
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(n)
        
        step_size = 0.2 * temp * 10 + 0.01
        perturbation = rng.randn(2) * step_size
        
        new_point = current_points[idx].copy()
        new_point += perturbation
        
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        if use_smooth:
            trial_score = smooth_min_max_ratio(trial_points, beta)
        else:
            trial_score = calculate_score(trial_points)
            
        if trial_score >= current_score:
            current_points = trial_points
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - trial_score
            if delta < 1e-10:
                current_points = trial_points
                current_score = trial_score
            else:
                if rng.rand() < np.exp(-delta / (temp * 10 + 1e-10)):
                    current_points = trial_points
                    current_score = trial_score
        
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
            
        if time.time() - start_time > time_limit * 0.95:
            break

    final_score = calculate_score(best_points)
    return best_points, final_score

def nelder_mead_polish(points, time_limit, seed=None):
    """
    Use Nelder-Mead on the smooth surrogate to polish the points.
    """
    n = points.shape[0]
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -smooth_min_max_ratio(pts, beta=0.05)
        
    start_time = time.time()
    best_pts = points.copy()
    best_score = calculate_score(best_pts)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10})
        
        if res.success or calculate_score(res.x.reshape(n,2)) > best_score:
            trial_pts = res.x.reshape(n,2)
            trial_score = calculate_score(trial_pts)
            if trial_score > best_score:
                best_pts = trial_pts
                best_score = trial_score
    except Exception:
        pass
        
    return best_pts, best_score

def optimize_geometry(points, time_limit, seed=None):
    """
    A structured optimizer combining local search and Nelder-Mead polish.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    
    while time.time() - start_time < time_limit - 5:
        chunk = min(remaining_time_calc(start_time, time_limit) - 2, 20)
        
        if chunk < 2:
            break
            
        # Phase 1: Smooth local search
        res_points, res_score = local_search(current_points, chunk, seed=seed, use_smooth=True, beta=0.1)
        if res_score > best_score:
            best_score = res_score
            best_points = res_points.copy()
            current_points = best_points.copy()
            
        # Phase 2: Exact local search
        chunk2 = min(remaining_time_calc(start_time, time_limit) - 2, 10)
        if chunk2 > 2:
            res_points, res_score = local_search(current_points, chunk2, seed=seed+1, use_smooth=False)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 3: L-BFGS-B with complex representation
        chunk3 = min(remaining_time_calc(start_time, time_limit) - 2, 15)
        if chunk3 > 2:
            res_points, res_score = lbfgsb_optimize(current_points, chunk3, beta=0.1, seed=seed+2)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 4: Nelder-Mead polish
        chunk4 = min(remaining_time_calc(start_time, time_limit) - 2, 10)
        if chunk4 > 2:
            res_points, res_score = nelder_mead_polish(current_points, chunk4, seed=seed+3)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Perturb for next restart
        num_perturb = rng.randint(1, 4)
        indices = rng.choice(n, num_perturb, replace=False)
        for idx in indices:
            current_points[idx] += rng.randn(2) * 0.2
            
    return best_points, best_score

def remaining_time_calc(start_time, total_limit):
    return total_limit - (time.time() - start_time)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 620
    
    candidates = []
    
    # Candidate 1: 4x4 Square Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(grid_points)
    
    # Candidate 3: Points on a circle + inner points
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi/8, 2*np.pi, 8, endpoint=False)
    r1 = 1.0
    r2 = 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    circle_points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    candidates.append(circle_points)
    
    # Candidate 4: Random points in a disk
    for i in range(5):
        rng_init = np.random.RandomState(42 + i)
        angles = rng_init.rand(n) * 2 * np.pi
        radii = np.sqrt(rng_init.rand(n)) * 1.5
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        candidates.append(np.column_stack((x, y)))
    
    # Candidate 5: Hexagonal lattice points (first 16 in a hex grid)
    hex_grid = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3)/2 * r
            hex_grid.append([x, y])
    hex_grid = np.array(hex_grid)
    dists_to_origin = np.linalg.norm(hex_grid, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_candidate = hex_grid[idx]
    candidates.append(hex_candidate)
    
    # Candidate 6: Symmetric configuration with 1 center + 5 rings
    # 1 center, 6 at r=1, 9 at r=sqrt(3)
    center = np.array([0.0, 0.0])
    ring1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
    r2 = np.sqrt(3)
    ring2 = r2 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    symmetric_points = np.vstack([center, ring1, ring2])
    candidates.append(symmetric_points)
    
    best_points = None
    best_score = 0
    
    num_candidates = len(candidates)
    time_per_candidate = 45
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 20:
            break
            
        cand = cand.astype(float)
        
        res_points, res_score = optimize_geometry(cand, time_per_candidate, seed=1000+i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    if best_points is not None and time.time() < global_deadline - 10:
        remaining_time = global_deadline - 10 - time.time()
        if remaining_time > 20:
            res_points, res_score = optimize_geometry(best_points, remaining_time, seed=9999)
            if res_score > best_score:
                best_points = res_points
                best_score = res_score

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
        
    return best_points

# EVOLVE-BLOCK-END
