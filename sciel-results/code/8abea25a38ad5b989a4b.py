"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_score(pts):
    """Compute the packing score for given points."""
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r


def _soft_min_obj_and_grad(pts, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    Maximizes the minimum of (pairwise distances / 2) and (boundary distances).
    """
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - radii
    
    # Combine all constraints into a single array
    # Pairwise: dists[i,j] for i<j
    pairwise_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_dists.append(dists[i, j])
    pairwise_dists = np.array(pairwise_dists)
    
    all_dists = np.concatenate([pairwise_dists, boundary_dists])
    
    # Soft min using log-sum-exp: soft_min(x) = -log(sum(exp(-x/beta)) * beta)
    scaled = -all_dists / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -beta * log_sum_exp
    
    # Gradient
    grad = np.zeros_like(pts)
    
    # Gradient for pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 1e-10:
                continue
            weight = np.exp(-d / beta - log_sum_exp)
            # d/d(pts[i]) of dist(i,j) = (pts[i] - pts[j]) / d
            grad[i] += weight * (pts[i] - pts[j]) / d
            # d/d(pts[j]) of dist(i,j) = (pts[j] - pts[i]) / d
            grad[j] += weight * (pts[j] - pts[i]) / d
    
    # Gradient for boundary distances
    for i in range(n):
        r = radii[i]
        if r < 1e-10:
            continue
        weight = np.exp(-boundary_dists[i] / beta - log_sum_exp)
        # d/d(pts[i]) of (1 - |pts[i]|) = -pts[i] / |pts[i]|
        grad[i] -= weight * pts[i] / r
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min_val, -grad


def _hex_lattice_init(n_points, n_rings=5):
    """Generate hexagonal lattice points within unit circle."""
    points = [np.array([0.0, 0.0])]
    
    ring_radii = []
    for ring in range(1, n_rings + 1):
        ring_radii.append(ring * 0.2)
    
    for ring, r in enumerate(ring_radii, 1):
        num_points = 6 * ring
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        for angle in angles:
            pt = np.array([r * np.cos(angle), r * np.sin(angle)])
            points.append(pt)
    
    points = np.array(points)
    # Keep only points within unit circle
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    points = points[radii <= 1.0]
    
    # Select n_points closest to center
    if len(points) >= n_points:
        radii = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(radii)[:n_points]
        return points[idx]
    else:
        # Pad with random points if needed
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1)
            points = np.vstack([points, np.array([r * np.cos(angle), r * np.sin(angle)])])
        return points


def _random_init(n_points, seed=None):
    """Generate random initial points."""
    if seed is not None:
        np.random.seed(seed)
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points))
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _optimize_from_start(pts0, beta_start=0.1, beta_end=0.01, max_iter=2000, time_budget=60.0):
    """Optimize from a starting point with annealing beta."""
    start_time = time.time()
    pts = pts0.copy()
    best_pts = pts0.copy()
    best_score = _compute_score(pts0)
    
    # Anneal beta
    n_steps = 20
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
        
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            result = minimize(
                lambda x: obj_and_grad(x)[0],
                pts,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter // n_steps, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts = result.x
            
            # Project points back inside unit circle
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            mask = radii > 1.0
            if np.any(mask):
                pts[mask] /= radii[mask, np.newaxis]
            
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    return best_pts, best_score


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n_points = N
    best_pts = None
    best_score = -1.0
    
    # Try multiple initialization strategies
    initializations = []
    
    # 1. Hexagonal lattice
    initializations.append(_hex_lattice_init(n_points, n_rings=5))
    
    # 2. Ring-based initialization (1+6+12+18+13 structure)
    ring_structure = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    ring_pts = []
    for count, r in zip(ring_structure, ring_radii):
        if count == 1:
            ring_pts.append(np.array([0.0, 0.0]))
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for angle in angles:
                ring_pts.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    ring_init = np.array(ring_pts[:n_points])
    initializations.append(ring_init)
    
    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1024]:
        initializations.append(_random_init(n_points, seed=seed))
    
    # 4. Golden angle spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    spiral_pts = []
    for i in range(n_points):
        r = np.sqrt((i + 0.5) / n_points)
        theta = i * golden_angle
        spiral_pts.append(np.array([r * np.cos(theta), r * np.sin(theta)]))
    initializations.append(np.array(spiral_pts))
    
    # Optimize from each initialization
    for idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        if remaining < 5.0:
            break
        
        # Allocate time based on remaining budget
        time_budget = min(25.0, remaining * 0.8)
        
        try:
            pts, score = _optimize_from_start(
                init_pts, 
                beta_start=0.15, 
                beta_end=0.005, 
                max_iter=1500,
                time_budget=time_budget
            )
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: _soft_min_obj_and_grad(x, 0.001)[0],
                best_pts,
                jac=lambda x: _soft_min_obj_and_grad(x, 0.001)[1],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished_pts = result.x
            
            # Project inside unit circle
            radii = np.sqrt(np.sum(polished_pts ** 2, axis=1))
            mask = radii > 1.0
            if np.any(mask):
                polished_pts[mask] /= radii[mask, np.newaxis]
            
            polished_score = _compute_score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                
        except Exception:
            pass
    
    # If no valid result found, fall back to hexagonal lattice
    if best_pts is None:
        best_pts = _hex_lattice_init(n_points, n_rings=6)
    
    return best_pts
# EVOLVE-BLOCK-END
