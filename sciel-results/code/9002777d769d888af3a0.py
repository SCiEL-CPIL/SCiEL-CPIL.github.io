"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the score: min of pairwise distances/2 and container margins."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Container margins
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = 1.0 - np.max(norms)
    return min(min_pair, min_margin)

def _hexagonal_initial(n_points):
    """Generate initial points on a hexagonal lattice inside unit circle."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    k = int(1.0 / spacing) + 3
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) < n_points:
        # Fallback: uniform random
        points = np.random.uniform(-0.8, 0.8, size=(n_points, 2))
    else:
        points = points[np.argsort(radius, kind="stable")[:n_points]]
    return points.copy()

def _neg_score_smooth(points, alpha=50.0):
    """Smooth surrogate: minimize -log-sum-exp of (min_pair, min_margin) scaled."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Container margins
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = 1.0 - np.max(norms)
    
    # Use log-sum-exp for smooth min
    vals = np.array([min_pair, min_margin])
    m = np.max(vals)
    lse = m + np.log(np.sum(np.exp(-alpha * (vals - m)))) / alpha
    return -lse

def _neg_score_exact(points):
    """Exact negative score for final evaluation."""
    return -_compute_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s to leave margin for 150s limit
    
    n_points = N
    best_points = None
    best_score = -np.inf
    
    # Initial hexagonal lattice
    points = _hexagonal_initial(n_points)
    score = _compute_score(points)
    best_points = points.copy()
    best_score = score
    
    # Simulated annealing with smooth objective
    np.random.seed(42)
    
    T = 0.01  # initial temperature
    T_min = 0.0001
    cooling = 0.995
    max_iter = 5000
    
    current_points = points.copy()
    current_score = score
    
    for iteration in range(max_iter):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb
        step_size = 0.02 * T / 0.01
        perturbed = current_points + np.random.normal(0, step_size, size=current_points.shape)
        
        # Ensure points stay in reasonable bounds
        norms = np.sqrt(np.sum(perturbed**2, axis=1))
        mask = norms > 0.95
        if np.any(mask):
            perturbed[mask] = 0.95 * perturbed[mask] / norms[mask, np.newaxis]
        
        new_score = _compute_score(perturbed)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score > current_score:
            current_points = perturbed
            current_score = new_score
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_points = perturbed
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    # Polish with L-BFGS-B using smooth objective
    if time.time() < deadline - 10.0:
        flat = best_points.flatten()
        def obj(x):
            pts = x.reshape(-1, 2)
            return _neg_score_smooth(pts, alpha=100.0)
        
        res = minimize(obj, flat, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success or res.fun < _neg_score_smooth(best_points, alpha=100.0):
            polished = res.x.reshape(-1, 2)
            pol_score = _compute_score(polished)
            if pol_score > best_score:
                best_points = polished
                best_score = pol_score
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        flat = best_points.flatten()
        def obj_exact(x):
            pts = x.reshape(-1, 2)
            return _neg_score_exact(pts)
        
        # Use multiple restarts for final polish
        for restart in range(3):
            if time.time() > deadline - 2.0:
                break
            if restart == 0:
                x0 = flat
            else:
                x0 = flat + np.random.normal(0, 0.001, size=flat.shape)
            
            res = minimize(obj_exact, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < -best_score:
                final_pts = res.x.reshape(-1, 2)
                final_score = _compute_score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
                    best_score = final_score
    
    # Verify validity
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    if np.any(norms > 1.0):
        # Clamp to boundary
        best_points = best_points / np.maximum(norms, 1e-10)[:, np.newaxis] * 0.999
    
    return best_points
# EVOLVE-BLOCK-END
