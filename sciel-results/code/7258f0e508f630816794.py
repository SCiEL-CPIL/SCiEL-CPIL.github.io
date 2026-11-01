import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score: max/min squared distance."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def normalize_points(points):
    """Center and scale points to have unit mean squared norm."""
    pts = points - np.mean(points, axis=0)
    norm = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if norm < 1e-15:
        return pts
    return pts / norm

def objective_smooth(x, tau=10.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    return np.log(soft_max) - np.log(soft_min)

def objective_epigraph(x, tau=50.0):
    """
    Epigraph-style smooth objective.
    Introduces t as the log-ratio, and smooth constraints.
    x is [points (N*D), t]
    """
    pts = x[:N*D].reshape(N, D)
    t = x[N*D]
    
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_dists = np.log(sq_dists)
    
    # Smooth max of (log_dists - t/2) and smooth min of (log_dists + t/2)
    # We want: max_i (log_dists_i) <= t/2 and min_i (log_dists_i) >= -t/2
    # So: max_i (log_dists_i - t/2) <= 0 and max_i (-log_dists_i - t/2) <= 0
    
    # Smooth max of (log_dists - t/2)
    shifted_max = log_dists - t/2
    m = np.max(shifted_max)
    smooth_max_1 = m + np.log(np.sum(np.exp(tau * (shifted_max - m)))) / tau
    
    # Smooth max of (-log_dists - t/2)
    shifted_min = -log_dists - t/2
    m2 = np.max(shifted_min)
    smooth_max_2 = m2 + np.log(np.sum(np.exp(tau * (shifted_min - m2)))) / tau
    
    # Penalty for violating constraints
    penalty = np.maximum(0, smooth_max_1)**2 + np.maximum(0, smooth_max_2)**2
    
    return t + penalty

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates (normalized)
    candidates = []
    
    # 1. Fibonacci sphere points (normalized)
    fib_pts = fibonacci_sphere(N)
    fib_pts_norm = normalize_points(fib_pts)
    candidates.append(fib_pts_norm.flatten())
    
    # 2. Random points on sphere with different seeds (normalized)
    for seed in range(8):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        pts_norm = normalize_points(pts)
        candidates.append(pts_norm.flatten())
        
        if time.time() - start_time > timeout - 30:
            break
    
    # 3. Random points in ball (normalized)
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        pts_norm = normalize_points(pts)
        candidates.append(pts_norm.flatten())
        
        if time.time() - start_time > timeout - 30:
            break
    
    # 4. Perturbed Fibonacci (normalized)
    for _ in range(3):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.3
        perturbed_norm = normalize_points(perturbed)
        candidates.append(perturbed_norm.flatten())
    
    # 5. Grid-based initialization (normalized)
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    grid_pts_arr = np.array(final_grid, dtype=float)
    grid_pts_arr = grid_pts_arr - np.mean(grid_pts_arr, axis=0)
    grid_pts_norm = normalize_points(grid_pts_arr)
    candidates.append(grid_pts_norm.flatten())
    
    # 6. Perturbed grid (normalized)
    for _ in range(3):
        perturbed = grid_pts_arr + rng.normal(size=(N, D)) * 0.2
        perturbed_norm = normalize_points(perturbed)
        candidates.append(perturbed_norm.flatten())
    
    # 7. Icosahedron-based: 12 vertices of icosahedron + 2 center-ish points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    # Add 2 more points near center but offset
    extra1 = np.array([0, 0, 0]) + rng.normal(size=D) * 0.1
    extra2 = np.array([0, 0, 0]) - rng.normal(size=D) * 0.1
    ico_pts = np.vstack([ico_verts, extra1.reshape(1, -1), extra2.reshape(1, -1)])
    ico_pts_norm = normalize_points(ico_pts)
    candidates.append(ico_pts_norm.flatten())
    
    # 8. Perturbed icosahedron
    for _ in range(2):
        perturbed = ico_pts + rng.normal(size=(N, D)) * 0.2
        perturbed_norm = normalize_points(perturbed)
        candidates.append(perturbed_norm.flatten())
    
    # Optimization parameters
    # Aggressive tau schedule for better approximation of nonsmooth max/min
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
    maxiter_schedule = [300, 400, 500, 600, 800]
    
    # Optimization loop over candidates
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout - 5:
                    break
                
                obj = partial(objective_smooth, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                # Use L-BFGS-B for the main optimization
                res = minimize(
                    obj,
                    x_curr,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            
            # Final high-precision polish with careful L-BFGS-B on exact-like objective
            if time.time() - start_time < timeout - 5.0:
                try:
                    # Use very high tau but with careful settings
                    obj_final = partial(objective_smooth, tau=10000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='L-BFGS-B',
                        options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                except Exception:
                    pass
            
            # Evaluate the true non-smooth score on normalized points
            pts = x_curr.reshape(N, D)
            pts_norm = normalize_points(pts)
            true_s = raw_score(pts_norm)
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts_norm.copy()
                
        except Exception:
            pass
    
    # If optimization didn't find anything better, fall back to the best candidate
    if best_points is None:
        # Try all candidates directly
        for init_x in candidates:
            pts = init_x.reshape(N, D)
            pts_norm = normalize_points(pts)
            s = raw_score(pts_norm)
            if np.isfinite(s) and s < best_score:
                best_score = s
                best_points = pts_norm.copy()
        
        if best_points is None:
            best_points = fib_pts_norm
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
