import numpy as np
import time
from scipy.optimize import minimize
import itertools
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute the true (non-smooth) score."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) <= 1e-12:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_log_ratio(x, tau=50.0):
    """
    Log-domain smooth approximation of log(max/min) of squared distances.
    This is better conditioned than the ratio directly.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    
    # Soft max in log domain
    m_max = np.max(log_dists)
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min in log domain
    m_min = np.min(log_dists)
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Return log ratio (we'll exponentiate for comparison, but optimize in log space)
    return soft_max_log - soft_min_log

def objective_epigraph(x, tau=50.0):
    """
    Epigraph-style smooth objective: minimize t subject to all sq_dists in [1/t, t].
    We use a smooth barrier: sum of soft-violations + log(t).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_dists = np.log(sq_dists)
    m_max = np.max(log_dists)
    m_min = np.min(log_dists)
    
    # Soft max and soft min
    exp_max_terms = np.exp(tau * (log_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (log_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # The log ratio
    log_ratio = soft_max_log - soft_min_log
    
    # Add a small penalty for very small distances to avoid degenerate solutions
    min_penalty = 0.0
    if m_min < -20:
        min_penalty = (m_min + 20) ** 2
    
    return log_ratio + min_penalty

def fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * radius

def icosahedron_points(radius=1.0):
    """Generate 12 icosahedron vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * radius
    return pts

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        s = raw_score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    # Initial candidates
    # 1. The grid from the current solution
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    # 2. Fibonacci sphere initialization
    fib_pts = fibonacci_sphere(N, radius=2.0)
    
    # 3. Icosahedron-based: 12 icosahedron vertices + 2 extra points
    ico_pts = icosahedron_points(radius=2.0)
    # Add 2 points near the center-top and center-bottom to make 14
    extra1 = np.array([0, 0, 2.0])
    extra2 = np.array([0, 0, -2.0])
    ico_extended = np.vstack([ico_pts, extra1.reshape(1,3), extra2.reshape(1,3)])
    ico_extended = ico_extended - np.mean(ico_extended, axis=0)
    
    # 4. Random starts
    rng = np.random.default_rng(42)
    
    candidates = [init_grid.flatten(), fib_pts.flatten(), ico_extended.flatten()]
    
    num_random_starts = 20
    for i in range(num_random_starts):
        if time.time() - start_time > timeout:
            break
        # Random points in a sphere
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * rng.uniform(1.0, 3.0)
        candidates.append(pts.flatten())
    
    # 5. Perturbed versions for diversity
    for base in [init_grid, fib_pts, ico_extended]:
        for _ in range(3):
            if time.time() - start_time > timeout:
                break
            perturbed = base + rng.normal(size=(N, D)) * 0.15
            candidates.append(perturbed.flatten())
    
    # Evaluate initial candidates
    for cand in candidates:
        try_update(cand.reshape(N, D))
    
    # Tau schedule for annealing-like optimization
    tau_schedule = [10.0, 30.0, 100.0, 300.0, 800.0]
    maxiter_schedule = [150, 200, 300, 400, 500]
    
    # Optimization loop
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        x_curr = init_x.copy()
        
        try:
            for idx, tau in enumerate(tau_schedule):
                if time.time() - start_time > timeout:
                    break
                
                # Use log-ratio objective for better conditioning
                obj = partial(objective_log_ratio, tau=tau)
                maxiter = maxiter_schedule[idx]
                
                res = minimize(
                    obj, 
                    x_curr, 
                    method='L-BFGS-B', 
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
                    try_update(x_curr.reshape(N, D))
            
            # Final high-precision polish with epigraph objective
            if time.time() - start_time < timeout - 5.0:
                try:
                    obj_final = partial(objective_epigraph, tau=2000.0)
                    res_polish = minimize(
                        obj_final,
                        x_curr,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if res_polish.x is not None and np.isfinite(res_polish.fun):
                        x_curr = res_polish.x
                        try_update(x_curr.reshape(N, D))
                except Exception:
                    pass
            
            # Additional SLSQP polish
            if time.time() - start_time < timeout - 3.0:
                try:
                    obj_slsqp = partial(objective_log_ratio, tau=3000.0)
                    res_slsqp = minimize(
                        obj_slsqp,
                        x_curr,
                        method='SLSQP',
                        options={'maxiter': 200, 'ftol': 1e-14}
                    )
                    if res_slsqp.x is not None and np.isfinite(res_slsqp.fun):
                        x_curr = res_slsqp.x
                        try_update(x_curr.reshape(N, D))
                except Exception:
                    pass
                    
        except Exception:
            pass

    # If optimization didn't find anything better, fall back to the grid
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
