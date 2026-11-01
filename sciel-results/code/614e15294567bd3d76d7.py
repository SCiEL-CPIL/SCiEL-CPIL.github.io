"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Multi-start Nelder-Mead with negative min-distance objective.
    """
    
    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return min(np.sqrt(np.min(d2)) / 2, 1 - np.max(d_origin))
    
    def objective(x):
        """
        Negative of the minimum radius: we minimize -r to maximize r.
        Use a smooth approximation to help the optimizer.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Get all pairwise distances and boundary distances
        d_pairs = np.sqrt(d2)
        d_boundary = 1 - d_origin
        
        # We want to maximize min(all d_pairs/2, all d_boundary)
        # Use a smooth min: -log-sum-exp approximation
        # min(a,b) ≈ -log(exp(-a) + exp(-b)) for large negative values
        # Better: use a power-based smooth min
        alpha = 100.0  # sharpness
        
        # Combine all constraints into a single smooth min
        all_dists = np.concatenate([d_pairs.flatten(), d_boundary])
        all_dists = np.maximum(all_dists, 1e-10)
        
        # Smooth min via log-sum-exp
        log_sum_exp = np.log(np.sum(np.exp(-alpha * all_dists)))
        smooth_min = -log_sum_exp / alpha
        
        return -smooth_min
    
    def init_random(seed, center_bias=0.0):
        """Random initialization with optional center bias."""
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2*np.pi, N)
        # Sample radius with bias toward center for better packing
        r_raw = rng.uniform(0, 1, N)
        r = r_raw ** 0.7 * 0.92  # Slightly inside boundary
        pts = np.column_stack([r*np.cos(a), r*np.sin(a)])
        # Add small random perturbation
        pts += rng.uniform(-0.01, 0.01, (N, 2))
        return pts.flatten()
    
    def init_grid_perturbed(seed):
        """Initialize on a grid with perturbation."""
        rng = np.random.default_rng(seed)
        # Create a grid of points
        n_per_side = 8
        x_vals = np.linspace(-0.7, 0.7, n_per_side)
        y_vals = np.linspace(-0.7, 0.7, n_per_side)
        pts = []
        for x in x_vals:
            for y in y_vals:
                if x*x + y*y < 0.81:  # Inside radius 0.9
                    pts.append([x, y])
        # Shuffle and take first N
        rng.shuffle(pts)
        pts = np.array(pts[:N])
        # Add perturbation
        pts += rng.uniform(-0.05, 0.05, (N, 2))
        # Clip to stay inside
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.9
        if np.any(too_far):
            scale = 0.9 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    def init_concentric_rings(seed):
        """Initialize with concentric rings."""
        rng = np.random.default_rng(seed)
        pts = [[0, 0]]
        rings = [(6, 0.2), (12, 0.4), (18, 0.6), (14, 0.8)]
        for n, rad in rings:
            angles = 2*np.pi*np.arange(n)/n + rng.uniform(0, 2*np.pi)
            for a in angles:
                pts.append([rad*np.cos(a), rad*np.sin(a)])
        pts = np.array(pts[:N])
        pts += rng.uniform(-0.02, 0.02, (N, 2))
        dists = np.linalg.norm(pts, axis=1)
        too_far = dists > 0.92
        if np.any(too_far):
            scale = 0.92 / dists[too_far]
            pts[too_far] *= scale[:, None]
        return pts.flatten()
    
    # Multiple diverse initializations
    inits = [
        init_random(1),
        init_random(2),
        init_random(3),
        init_random(4),
        init_random(5),
        init_grid_perturbed(10),
        init_grid_perturbed(20),
        init_concentric_rings(30),
    ]
    
    best_x, best_r = None, -1
    
    # First pass: coarse optimization
    for i, x0 in enumerate(inits):
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'maxfev': 5000,
                                   'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            r = get_radius(res.x)
            if r > best_r:
                best_r, best_x = r, res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        return init_random(1).reshape(N, 2)
    
    # Second pass: fine optimization with tighter tolerances
    try:
        res2 = minimize(objective, best_x, method='Nelder-Mead',
                       options={'maxiter': 5000, 'maxfev': 8000,
                                'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        r2 = get_radius(res2.x)
        if r2 > best_r:
            best_x = res2.x.copy()
            best_r = r2
    except Exception:
        pass
    
    # Third pass: very fine optimization
    try:
        res3 = minimize(objective, best_x, method='Nelder-Mead',
                       options={'maxiter': 3000, 'maxfev': 5000,
                                'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        r3 = get_radius(res3.x)
        if r3 > best_r:
            best_x = res3.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
