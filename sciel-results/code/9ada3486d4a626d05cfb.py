"""Packing 200 equal circles in a unit square [-1/2,1/2]^2."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 200
SIDE = 1.0
HALF = 0.5
TIME_LIMIT = 130.0  # safety margin below 150s kill

def _compute_score(pts):
    """Compute the packing score (max common radius) for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n != N:
        return 0.0
    # Check finite
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts < -HALF) or np.any(pts > HALF):
        return 0.0
    # Pairwise distances
    # Use chunked computation to avoid memory issues
    min_dist = np.inf
    for i in range(n):
        diff = pts[i+1:] - pts[i]
        d = np.sqrt(np.sum(diff*diff, axis=1))
        min_dist = min(min_dist, np.min(d))
    # Distance to boundary
    boundary_dists = np.min(np.stack([
        HALF - np.abs(pts[:, 0]),
        HALF - np.abs(pts[:, 1])
    ], axis=1), axis=1)
    min_boundary = np.min(boundary_dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _pairwise_min_dist(pts):
    """Compute minimum pairwise distance."""
    n = len(pts)
    min_dist = np.inf
    for i in range(n):
        diff = pts[i+1:] - pts[i]
        d = np.sqrt(np.sum(diff*diff, axis=1))
        min_dist = min(min_dist, np.min(d))
    return min_dist

def _boundary_min(pts):
    """Compute minimum distance to boundary."""
    return np.min(np.stack([
        HALF - np.abs(pts[:, 0]),
        HALF - np.abs(pts[:, 1])
    ], axis=1), axis=1)

def _objective_neg_r(pts_flat):
    """Negative radius objective for minimization."""
    pts = pts_flat.reshape(-1, 2)
    # If any point is outside, penalize heavily
    if np.any(pts < -HALF - 1e-10) or np.any(pts > HALF + 1e-10):
        return 1e10
    min_dist = _pairwise_min_dist(pts)
    min_bnd = np.min(_boundary_min(pts))
    r = min(min_dist / 2.0, min_bnd)
    return -r

def _gradient_neg_r(pts_flat):
    """Gradient of negative radius (subgradient approximation)."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    min_dist = _pairwise_min_dist(pts)
    min_bnd = np.min(_boundary_min(pts))
    r = min(min_dist / 2.0, min_bnd)
    
    grad = np.zeros_like(pts)
    
    if min_dist / 2.0 <= min_bnd:
        # Constrained by pairwise distance
        # Find the pair achieving min_dist
        for i in range(n):
            diff = pts[i+1:] - pts[i]
            d = np.sqrt(np.sum(diff*diff, axis=1))
            idx = np.argmin(d)
            if abs(d[idx] - min_dist) < 1e-12:
                j = i + 1 + idx
                direction = (pts[i] - pts[j]) / min_dist
                grad[i] += direction
                grad[j] -= direction
                break
    else:
        # Constrained by boundary
        bnd_mins = _boundary_min(pts)
        idx = np.argmin(bnd_mins)
        # Determine which boundary
        x, y = pts[idx]
        if abs(HALF - x) == bnd_mins[idx]:
            grad[idx, 0] = -1.0 if x > 0 else 1.0
        elif abs(HALF + x) == bnd_mins[idx]:
            grad[idx, 0] = 1.0 if x < 0 else -1.0
        else:
            grad[idx, 1] = -1.0 if y > 0 else 1.0
    
    # Scale: d(-r)/d(pts) = -grad(r)/2 for pair constraint
    # Actually r = min_dist/2, so d(-r)/d(pts) = -0.5 * d(min_dist)/d(pts)
    if min_dist / 2.0 <= min_bnd:
        grad *= 0.5
    else:
        grad *= 1.0
    
    return -grad.ravel()

def _hexagonal_init(n):
    """Create a hexagonal packing initialization."""
    pts = []
    # Hexagonal lattice
    r = 0.035  # approximate radius
    dx = 2*r
    dy = 2*r * np.sqrt(3)/2
    
    row = 0
    while len(pts) < n:
        offset = 0 if row % 2 == 0 else r
        x = -HALF + r + offset
        while x < HALF - r + 1e-10:
            y = -HALF + r + row * dy
            if y < HALF - r + 1e-10:
                pts.append([x, y])
            x += dx
        row += 1
    
    pts = np.array(pts[:n])
    return pts

def _grid_init(n):
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    # Try to make it as square as possible
    best_m = m
    best_err = abs(m*m - n)
    for mm in range(m-3, m+4):
        if mm < 1:
            continue
        err = abs(mm*mm - n)
        if err < best_err:
            best_err = err
            best_m = mm
    m = best_m
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=42):
    """Random initialization inside the square."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-HALF + 0.04, HALF - 0.04, size=(n, 2))
    return pts

def _force_relaxation(pts, n_iter=5000, cooling=0.9995):
    """Simple force-directed relaxation to improve packing."""
    pts = pts.copy()
    n = len(pts)
    T = 0.01
    best_score = _compute_score(pts)
    
    for it in range(n_iter):
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        for i in range(n):
            diff = pts[i+1:] - pts[i]
            d = np.sqrt(np.sum(diff*diff, axis=1))
            d[d < 1e-10] = 1e-10
            # Force: push apart if too close
            overlap = np.maximum(0, 0.072 - d)  # target min dist ~ 2*0.036
            f = overlap / d[:, None] * diff
            forces[i] -= np.sum(f, axis=0)
            forces[i+1:] += np.sum(f, axis=0)
        
        # Boundary force
        for i in range(n):
            for axis in range(2):
                if pts[i, axis] > HALF - 0.037:
                    forces[i, axis] -= (pts[i, axis] - (HALF - 0.037)) * 10
                elif pts[i, axis] < -HALF + 0.037:
                    forces[i, axis] += (-HALF + 0.037 - pts[i, axis]) * 10
        
        # Update
        pts += T * forces
        # Clamp to square
        pts = np.clip(pts, -HALF, HALF)
        
        T *= cooling
        
        if it % 500 == 0:
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
    
    return pts

def _slsqp_polish(pts, time_budget):
    """Polish using SLSQP with inequality constraints."""
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # Define bounds
    bounds = [(-HALF, HALF)] * (2 * n)
    
    # We'll use a simplified approach: optimize to maximize min distance
    # Use scipy minimize with method SLSQP
    
    def obj(x):
        p = x.reshape(-1, 2)
        # Penalize if outside
        penalty = 0
        if np.any(p < -HALF) or np.any(p > HALF):
            penalty = 1e6 * (np.max(np.abs(p) - HALF) + 1)
        min_d = _pairwise_min_dist(p)
        min_b = np.min(_boundary_min(p))
        r = min(min_d / 2.0, min_b)
        return -r + penalty
    
    def grad(x):
        p = x.reshape(-1, 2)
        min_d = _pairwise_min_dist(p)
        min_b = np.min(_boundary_min(p))
        g = np.zeros(2*n)
        if min_d / 2.0 <= min_b:
            # Find closest pair
            for i in range(n):
                diff = p[i+1:] - p[i]
                d = np.sqrt(np.sum(diff*diff, axis=1))
                idx = np.argmin(d)
                if abs(d[idx] - min_d) < 1e-12:
                    j = i + 1 + idx
                    direction = (p[i] - p[j]) / max(min_d, 1e-10)
                    g[2*i:2*i+2] = 0.5 * direction
                    g[2*j:2*j+2] = -0.5 * direction
                    break
        else:
            bnd_mins = _boundary_min(p)
            idx = np.argmin(bnd_mins)
            x_, y_ = p[idx]
            if abs(HALF - x_) <= abs(HALF + x_) and abs(HALF - x_) <= abs(HALF - y_) and abs(HALF - x_) <= abs(HALF + y_):
                g[2*idx] = -0.5
            elif abs(HALF + x_) <= abs(HALF - x_) and abs(HALF + x_) <= abs(HALF - y_) and abs(HALF + x_) <= abs(HALF + y_):
                g[2*idx] = 0.5
            elif abs(HALF - y_) <= abs(HALF - x_) and abs(HALF - y_) <= abs(HALF + x_) and abs(HALF - y_) <= abs(HALF + y_):
                g[2*idx+1] = -0.5
            else:
                g[2*idx+1] = 0.5
        return g
    
    x0 = pts.ravel()
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad, bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            new_pts = res.x.reshape(-1, 2)
            if _compute_score(new_pts) > _compute_score(pts):
                pts = new_pts
    except Exception:
        pass
    
    return pts

def _lbfgs_polish(pts, time_budget):
    """Polish using L-BFGS-B."""
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    bounds = [(-HALF, HALF)] * (2 * n)
    
    def obj(x):
        p = x.reshape(-1, 2)
        penalty = 0
        if np.any(p < -HALF) or np.any(p > HALF):
            penalty = 1e6 * (np.max(np.abs(p) - HALF) + 1)
        min_d = _pairwise_min_dist(p)
        min_b = np.min(_boundary_min(p))
        r = min(min_d / 2.0, min_b)
        return -r + penalty
    
    x0 = pts.ravel()
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000})
        if res.x is not None and np.all(np.isfinite(res.x)):
            new_pts = res.x.reshape(-1, 2)
            if _compute_score(new_pts) > _compute_score(pts):
                pts = new_pts
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal
    hex_pts = _hexagonal_init(N)
    inits.append(hex_pts)
    
    # Grid
    grid_pts = _grid_init(N)
    inits.append(grid_pts)
    
    # Random with different seeds
    for seed in [1, 2, 3, 42]:
        rand_pts = _random_init(N, seed=seed)
        inits.append(rand_pts)
    
    best_pts = None
    best_score = 0.0
    
    for init_pts in inits:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Force relaxation
        relaxed = _force_relaxation(init_pts, n_iter=3000)
        score = _compute_score(relaxed)
        
        if score > best_score:
            best_score = score
            best_pts = relaxed.copy()
        
        # L-BFGS polish
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        polished = _lbfgs_polish(relaxed, time_budget=20)
        score = _compute_score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Final high-precision polish
    if best_pts is not None and time.time() - start_time < TIME_LIMIT - 5:
        final = _lbfgs_polish(best_pts, time_budget=10)
        score = _compute_score(final)
        if score > best_score:
            best_pts = final
            best_score = score
    
    # Ensure validity
    if best_pts is None:
        # Fallback: grid
        best_pts = _grid_init(N)
    
    # Final clamp
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
