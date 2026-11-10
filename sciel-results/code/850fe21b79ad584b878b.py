"""Improved circle packing: hexagonal lattice init + gradient-free local optimization."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _score(pts):
    """Compute the packing radius (min of half-pair-dist and 1-norm)."""
    n = len(pts)
    # Pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    # Boundary
    norms = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_bound = (1.0 - norms).min()
    return min(min_pair, min_bound)


def _grad_free_optimize(pts, time_budget=120.0, tol=1e-12):
    """
    Local optimization using a variant of the repulsion-based approach.
    We maximize the minimum of (pairwise_dist/2, 1-|c|).
    Use a soft-min surrogate for smoothness, then polish.
    
    Actually, let's use a simpler approach: iterative pairwise separation + boundary push.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Precompute for efficiency
    deadline = time.time() + time_budget
    
    # We'll use a simple iterative scheme:
    # For each pair that is too close, push them apart.
    # For each point too close to boundary, pull it inward.
    # Repeat until convergence or time limit.
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Use a cooling schedule
    step_size = 0.01
    for iteration in range(100000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute all pairwise distances and boundary distances
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
        norms = np.sqrt(np.sum(pts ** 2, axis=-1))
        bound_dist = 1.0 - norms
        
        current_score = min(d.min() / 2.0, bound_dist.min())
        
        if current_score > best_score + 1e-15:
            best_score = current_score
            best_pts = pts.copy()
        
        if abs(current_score - best_score) < tol and iteration > 100:
            break
        
        # Find the most violated constraint
        half_d = d / 2.0
        np.fill_diagonal(half_d, np.inf)
        
        min_half_d = half_d.min()
        min_bound = bound_dist.min()
        
        if min_half_d < min_bound:
            # Pairwise constraint is more violated
            i, j = np.unravel_index(half_d.argmin(), half_d.shape)
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < 1e-15:
                # Random perturbation
                diff = np.random.randn(2)
                dist = np.sqrt(np.sum(diff ** 2))
            diff /= dist
            # Push apart
            pts[i] += step_size * diff
            pts[j] -= step_size * diff
        else:
            # Boundary constraint is more violated
            i = bound_dist.argmin()
            norm_i = np.sqrt(np.sum(pts[i] ** 2))
            if norm_i < 1e-15:
                direction = np.random.randn(2)
                direction /= np.sqrt(np.sum(direction ** 2))
            else:
                direction = -pts[i] / norm_i
            pts[i] += step_size * direction
        
        # Cool down
        if iteration % 100 == 0:
            step_size *= 0.995
    
    return best_pts, best_score


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice inside the unit disk."""
    # Try different spacings
    best_pts = None
    best_count = 0
    
    for spacing_mult in np.linspace(0.8, 1.5, 20):
        # Estimate needed spacing
        # Area per point in hex lattice ~ (sqrt(3)/2) * s^2
        # Total area ~ pi * 1^2 = pi
        # n_points * (sqrt(3)/2) * s^2 ~ pi
        # s ~ sqrt(2*pi / (n_points * sqrt(3)))
        s = spacing_mult * np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
        
        k = int(1.0 / s) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    rows.append((x, y))
        
        pts = np.array(rows)
        if len(pts) >= n_points and len(pts) > best_count:
            best_pts = pts
            best_count = len(pts)
    
    if best_pts is None:
        # Fallback: simple grid
        s = 0.05
        k = int(1.0 / s) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * s
                y = j * s
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    rows.append((x, y))
        best_pts = np.array(rows)
    
    # Select n_points points, preferring those closer to center for better packing
    norms = np.sqrt(np.sum(best_pts ** 2, axis=-1))
    idx = np.argsort(norms)[:n_points]
    return best_pts[idx]


def construct():
    start_time = time.time()
    deadline = start_time + 150.0  # Leave 30s margin for safety
    
    # Initial hexagonal lattice
    pts = _hex_lattice_init(N)
    
    # Ensure we have exactly N points
    if len(pts) < N:
        # Add random points
        while len(pts) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            pts = np.vstack([pts, [[r * np.cos(angle), r * np.sin(angle)]]])
    elif len(pts) > N:
        pts = pts[:N]
    
    # Local optimization
    pts, score = _grad_free_optimize(pts, time_budget=130.0)
    
    # Verify validity
    if not np.all(np.isfinite(pts)):
        # Fallback to hex lattice
        pts = _hex_lattice_init(N)
    
    return pts
# EVOLVE-BLOCK-END
