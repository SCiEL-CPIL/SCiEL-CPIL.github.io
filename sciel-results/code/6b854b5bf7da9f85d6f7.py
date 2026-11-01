"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_init(N):
    """Generate hexagonal lattice points, select N closest to origin."""
    # Estimate spacing for ~N points in unit disk
    # Area of unit disk = pi, area per hex point = sqrt(3)/2 * s^2
    # N * sqrt(3)/2 * s^2 ~ pi => s ~ sqrt(2*pi/(N*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.95  # slightly tighter to ensure enough points
    
    best_points = None
    for attempt in range(20):
        k = int(1.0 / s) + 3
        # Generate hex lattice
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s / 2.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Select N points closest to origin
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            if best_points is None or len(idx) > len(best_points):
                best_points = pts[idx]
        s *= 0.98
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    # Min distance between any pair / 2
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        # Compute all pairwise distances
        dx = pts[:, 0:1] - pts[0:1, 0]
        # More efficient: use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    
    # Min distance to boundary
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)
    
    return min(min_pair, min_boundary)

def _soft_min_value(pts, temperature=0.01):
    """
    Smooth surrogate for the min radius using log-sum-exp.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalently, minimize -r.
    
    Using soft-min: soft_min(x) = -T * log(sum(exp(-x_i/T)))
    """
    n = len(pts)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
    
    # Soft min via log-sum-exp (stable version)
    all_vals = np.clip(all_vals, -100, 100)  # avoid overflow
    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    soft_min = min_val - temperature * np.log(np.sum(exp_vals))
    
    return -soft_min  # We minimize the negative

def _soft_min_grad(pts, temperature=0.01):
    """Gradient of the soft-min objective."""
    n = len(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    boundary_vals = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    all_vals = np.concatenate([pair_vals.flatten(), boundary_vals])
    all_vals = np.clip(all_vals, -100, 100)
    
    min_val = np.min(all_vals)
    exp_vals = np.exp((all_vals - min_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    # Pairwise part
    pair_weights = weights[:n*(n-1)]
    pair_weights = pair_weights.reshape(n, n)
    # For each pair (i,j), gradient contribution to point i
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            if dists[i, j] > 1e-12:
                # d(pair_dist/2)/d(pts[i]) = (pts[i] - pts[j]) / (2 * pair_dist) * w
                grad[i] += (pts[i] - pts[j]) / (2.0 * dists[i, j]) * w * (-temperature)
    
    # Boundary part
    boundary_weights = weights[n*(n-1):]
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            # d(1 - r)/d(pts[i]) = -pts[i]/r
            grad[i] += -pts[i] / r * boundary_weights[i] * (-temperature)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Initial hexagonal packing
    pts = _hex_init(N)
    if pts is None or len(pts) != N:
        # Fallback: random initialization
        pts = np.random.randn(N, 2) * 0.8
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Try multiple temperatures for soft-min optimization
    temperatures = [0.005, 0.01, 0.02, 0.05]
    
    for temp in temperatures:
        if time.time() > deadline - 30.0:
            break
        
        # Use L-BFGS-B to minimize the soft-min objective
        def objective(x):
            p = x.reshape(N, 2)
            return _soft_min_value(p, temperature=temp)
        
        # Gradient
        def gradient(x):
            p = x.reshape(N, 2)
            return _soft_min_grad(p, temperature=temp).flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.success or result.fun < objective(x0):
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
                    pts = new_pts
        
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
    
    # Final polish: try to improve with smaller temperature
    if time.time() < deadline - 15.0:
        temp = 0.002
        def objective(x):
            p = x.reshape(N, 2)
            return _soft_min_value(p, temperature=temp)
        
        def gradient(x):
            p = x.reshape(N, 2)
            return _soft_min_grad(p, temperature=temp).flatten()
        
        x0 = pts.flatten()
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.success:
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Ensure validity: clip points inside unit circle
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists <= 1.0
    if not np.all(mask):
        # Project points outside the circle back inside
        for i in range(N):
            if dists[i] > 1.0:
                best_pts[i] = best_pts[i] / dists[i] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
