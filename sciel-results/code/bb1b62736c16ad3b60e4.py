import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 148.0  # Leave 2s margin for safety

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    n = len(pts)
    if n == 0:
        return 0.0
    
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_boundary = np.minimum(dx, dy).min()
    
    if r_boundary < 0:
        return 0.0  # Invalid, outside square

    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    
    return min(min_pair / 2.0, r_boundary)

def _objective_and_grad(x):
    """
    Negative radius objective with gradient.
    We minimize -r where r = min(min_pair/2, min_boundary)
    Using smooth approximation via log-sum-exp.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    r_b = np.minimum(dx, dy)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Use a soft min with high beta for sharpness
    beta = 100.0
    
    # Combine all constraints: pairwise distances / 2 and boundary distances
    # We want to maximize min of these, so minimize -min
    # Use log-sum-exp trick for -softmin
    
    # Pairwise: d_ij / 2
    pair_vals = dists / 2.0
    # Boundary: r_b
    # We need to compute -softmin(pair_vals, r_b)
    # = softmax(-pair_vals, -r_b) weighted sum
    
    # For stability, subtract max
    all_vals = np.concatenate([pair_vals[np.triu_indices(n, k=1)], r_b])
    
    # Actually, let's just compute the gradient numerically or use a simpler approach
    # Given time constraints, let's use scipy's minimize with a smooth objective
    
    # Simple smooth objective: -min + penalty for violations
    r_val = min(pair_vals.min(), r_b.min())
    
    # Gradient: we need d(-r)/dx
    # This is complex. Let's use finite differences or a different strategy.
    # For now, return a value and let scipy handle it, or use L-BFGS-B with bounds
    
    return -r_val

def _initial_hex_grid(offset_x=0.0, offset_y=0.0, spacing=None):
    """Generate a hexagonal grid of N points centered in the square."""
    if spacing is None:
        # Estimate spacing
        s = np.sqrt(2.0 / (N * np.sqrt(3)))
    else:
        s = spacing
    
    points = []
    max_idx = int(1.0 / (s * 0.866)) + 2
    
    for row in range(-max_idx, max_idx + 1):
        y = row * s * np.sqrt(3) / 2 + offset_y
        if y < -0.5 or y > 0.5:
            continue
        x_offset = 0.5 * s if row % 2 != 0 else 0.0
        for col in range(-max_idx, max_idx + 1):
            x = col * s + x_offset + offset_x
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
            
    points = np.array(points)
    if len(points) == 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Pad with random points near existing ones
        while len(points) < N:
            idx = np.random.randint(len(points))
            new_pt = points[idx] + np.random.normal(0, 0.01, 2)
            new_pt = np.clip(new_pt, -0.49, 0.49)
            points = np.vstack([points, new_pt])
        
    return points[:N]

def _initial_random():
    """Random initialization."""
    pts = np.random.uniform(-0.45, 0.45, (N, 2))
    return pts

def _optimize_with_scipy(pts, time_limit=20.0):
    """Use scipy L-BFGS-B to optimize."""
    x0 = pts.flatten()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return np.inf
        pts = x.reshape(-1, 2)
        r = _compute_radius(pts)
        return -r
    
    # L-BFGS-B doesn't support gradient easily without providing it
    # Use a simple gradient-free or provide numerical gradient
    # For speed, let's use Powell or Nelder-Mead if L-BFGS-B is too slow
    
    # Actually, let's try a different approach: local search with random restarts
    # or use the fact that this is a max-min problem
    
    # Given the complexity, let's stick with a simpler gradient approach
    # or use scipy's minimize with method='L-BFGS-B' and no gradient (numerical)
    
    result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if time.time() - start_time > time_limit:
        return pts, _compute_radius(pts)
    
    if result.success:
        new_pts = result.x.reshape(-1, 2)
        new_r = _compute_radius(new_pts)
        if new_r > _compute_radius(pts):
            return new_pts, new_r
    
    return pts, _compute_radius(pts)

def _local_search(pts, time_limit=10.0):
    """Simple local search with random perturbations."""
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Random perturbation
        scale = 0.01
        perturbation = np.random.normal(0, scale, (N, 2))
        new_pts = pts + perturbation
        new_pts[:, 0] = np.clip(new_pts[:, 0], -0.499, 0.499)
        new_pts[:, 1] = np.clip(new_pts[:, 1], -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce perturbation scale occasionally
            if np.random.random() < 0.1:
                scale *= 0.9
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    best_pts = None
    best_r = 0.0
    
    # Multi-start strategy
    # 1. Hexagonal grids with different offsets
    # 2. Random initializations
    # 3. Local search on each
    
    offsets = [
        (0.0, 0.0),
        (0.005, 0.0),
        (0.0, 0.005),
        (0.005, 0.005),
        (-0.005, 0.0),
        (0.0, -0.005),
        (0.01, 0.01),
        (-0.01, -0.01),
    ]
    
    # Try hexagonal grids
    for ox, oy in offsets:
        if time.time() > DEADLINE - 30:
            break
            
        pts = _initial_hex_grid(ox, oy)
        if len(pts) < N:
            continue
            
        # Local search
        pts, r = _local_search(pts, time_limit=15.0)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Try a few random starts if time permits
    if time.time() < DEADLINE - 20:
        for _ in range(3):
            if time.time() > DEADLINE - 10:
                break
            pts = _initial_random()
            pts, r = _local_search(pts, time_limit=10.0)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Final polish with smaller steps
    if best_pts is not None and time.time() < DEADLINE - 5:
        best_pts, best_r = _local_search(best_pts, time_limit=5.0)
    
    if best_pts is None:
        # Emergency fallback
        best_pts = _initial_hex_grid(0.0, 0.0)
    
    return best_pts
