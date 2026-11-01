import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # 1. Initial Hexagonal Lattice - better spacing estimate
    # Area per hex point = sqrt(3)/2 * s^2, N points in unit disk (area pi)
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi/(N*sqrt(3)))
    s_est = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    best_init = None
    for attempt in range(30):
        spacing = s_est * (1.0 - 0.02 * attempt)
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2 inside unit circle
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Select N points closest to origin
            idx = np.argsort(radius, kind="stable")[:N]
            if best_init is None:
                best_init = points[idx].copy()
            # Try tighter spacing
            continue
        if attempt >= 5:
            break
    
    if best_init is None:
        # Fallback: random points in disk
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        best_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    x0 = best_init.copy()
    
    def get_radius_candidates(x):
        """Returns array of 'radius candidates': boundary distances and half-pairwise distances."""
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        # Use cdist for pairwise distances (more efficient than broadcasting)
        d_pair = cdist(x, x) / 2.0
        # Remove diagonal
        np.fill_diagonal(d_pair, np.inf)
        return np.concatenate([dist_boundary, d_pair.flatten()])
    
    def soft_min_score(x, beta):
        """
        Smooth surrogate for -min(radius_candidates).
        soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
        We minimize this to maximize min(d).
        """
        d = get_radius_candidates(x)
        # Numerical stability: clip to avoid overflow
        d = np.clip(d, -50.0, 50.0)
        A = -beta * d
        max_A = np.max(A)
        log_sum_exp = max_A + np.log(np.sum(np.exp(A - max_A)))
        return log_sum_exp / beta
    
    def neg_soft_min(x, beta):
        return soft_min_score(x, beta)
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    current_x = x0.copy()
    best_x = x0.copy()
    
    def raw_score(x):
        x = x.reshape(-1, 2)
        d = get_radius_candidates(x)
        return np.min(d)
    
    best_raw = raw_score(current_x)
    
    # Annealing loop with multiple beta values
    betas = [3, 8, 15, 30, 60, 120]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(neg_soft_min, current_x, args=(beta,), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxfun': 5000})
            if res.fun < soft_min_score(current_x, beta):
                current_x = res.x
        except Exception:
            pass
        
        # Update best
        raw_s = raw_score(current_x)
        if raw_s > best_raw:
            best_raw = raw_s
            best_x = current_x.copy()
    
    # Final polish: Nelder-Mead on the raw nonsmooth objective
    if time.time() < deadline - 20:
        def neg_raw_score(x):
            return -raw_score(x)
        
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < -best_raw:
                best_x = res.x
                best_raw = -res.fun
        except Exception:
            pass
    
    # Another polish round if time permits
    if time.time() < deadline - 10:
        def neg_raw_score(x):
            return -raw_score(x)
        
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if res.fun < -best_raw:
                best_x = res.x
                best_raw = -res.fun
        except Exception:
            pass
    
    # Final validation: ensure all points are strictly inside the unit circle
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 0.9999
    if np.any(mask):
        # Scale points slightly inward
        scale = 0.9999 / norms[mask]
        best_x[mask] *= scale[:, np.newaxis]
    
    return best_x
