"""Rewritten program for packing 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the largest common radius for given centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    d_boundary = np.min(np.stack([0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])], axis=1))
    if np.min(d_boundary) < 0:
        return 0.0
    # Minimum pairwise distance / 2
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    d_pairs = np.min(d, axis=1)
    return float(min(np.min(d_boundary), np.min(d_pairs) / 2.0))


def _initialize_hexagonal():
    """Create a hexagonal lattice initialization for N circles."""
    # For 200 circles, use a hexagonal arrangement
    # Hexagonal packing has rows offset by half the spacing
    # Let's create a grid that approximates hexagonal packing
    
    # Target radius ~ 0.0366, so spacing ~ 0.0732
    # In a unit square, we can fit about 13-14 circles per row
    # Hexagonal: row spacing = spacing * sqrt(3)/2
    
    spacing = 0.0732
    row_spacing = spacing * np.sqrt(3) / 2
    
    pts = []
    y = -0.5 + row_spacing / 2
    row = 0
    while y <= 0.5:
        x_offset = (row % 2) * (spacing / 2)
        x = -0.5 + spacing / 2 + x_offset
        while x <= 0.5:
            if x + spacing / 2 <= 0.5:  # ensure circle fits
                pts.append([x, y])
            x += spacing
        y += row_spacing
        row += 1
    
    pts = np.array(pts)
    if len(pts) < N:
        # If we don't have enough, add more by adjusting
        # Pad with small random perturbations near existing points
        extra_needed = N - len(pts)
        for i in range(extra_needed):
            idx = i % len(pts)
            new_pt = pts[idx] + np.random.normal(0, 0.01, 2)
            new_pt = np.clip(new_pt, -0.49, 0.49)
            pts = np.vstack([pts, [new_pt]])
    
    # Shuffle and take N
    idx = np.random.permutation(len(pts))
    return pts[idx[:N]]


def _initialize_grid():
    """Simple grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _initialize_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _initialize_jittered_hex():
    """Hexagonal with small random jitter."""
    pts = _initialize_hexagonal()
    pts += np.random.normal(0, 0.005, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def _soft_min_objective(x, alpha):
    """
    Smooth surrogate for the max-min objective using log-sum-exp soft-min.
    We want to maximize min(pairwise distances/2, boundary distances).
    So we minimize -min(...) using a soft approximation.
    """
    p = x.reshape(-1, 2)
    p = np.clip(p, -0.5, 0.5)
    
    # Pairwise distances
    d = cdist(p, p)
    np.fill_diagonal(d, np.inf)
    d_pairs = np.min(d, axis=1) / 2.0  # radius from pairwise
    
    # Boundary distances
    d_boundary = np.min(np.stack([0.5 - np.abs(p[:, 0]), 0.5 - np.abs(p[:, 1])], axis=1))
    
    # All candidate radii
    all_d = np.concatenate([d_pairs, d_boundary])
    
    # Soft min via log-sum-exp
    # soft_min = -log(sum(exp(-alpha * d))) / alpha
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(all_d)
    exp_sum = np.sum(np.exp(-alpha * (all_d - min_val)))
    soft_min = min_val - np.log(exp_sum) / alpha
    
    return -soft_min


def _local_optimize(pts, time_limit, alpha_schedule=None):
    """
    Local optimization with multi-phase alpha annealing.
    Start with small alpha (smoother), increase to large alpha (sharper).
    """
    if alpha_schedule is None:
        alpha_schedule = [10, 50, 100, 200, 500]
    
    start = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for alpha in alpha_schedule:
        if time.time() - start > time_limit:
            break
        
        def obj(x):
            return _soft_min_objective(x, alpha)
        
        try:
            res = minimize(
                obj, best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if time.time() - start > time_limit:
                break
            
            candidate = res.x.reshape(-1, 2)
            r = _compute_radius(candidate)
            if r > best_r:
                best_r = r
                best_pts = candidate.copy()
        except Exception:
            pass
    
    return best_pts, best_r


def _polish_high_precision(pts, time_limit):
    """Final high-precision polish with very high alpha."""
    start = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for alpha in [1000, 2000, 5000]:
        if time.time() - start > time_limit:
            break
        
        def obj(x):
            return _soft_min_objective(x, alpha)
        
        try:
            res = minimize(
                obj, best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if time.time() - start > time_limit:
                break
            
            candidate = res.x.reshape(-1, 2)
            r = _compute_radius(candidate)
            if r > best_r:
                best_r = r
                best_pts = candidate.copy()
        except Exception:
            pass
    
    return best_pts, best_r


def _active_set_polish(pts, time_limit):
    """
    Polish by identifying the active constraints (closest pairs and boundary contacts)
    and optimizing only those points more aggressively.
    """
    start = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for iteration in range(5):
        if time.time() - start > time_limit:
            break
        
        # Identify active constraints
        d = cdist(best_pts, best_pts)
        np.fill_diagonal(d, np.inf)
        d_pairs = np.min(d, axis=1) / 2.0
        d_boundary = np.min(np.stack([0.5 - np.abs(best_pts[:, 0]), 0.5 - np.abs(best_pts[:, 1])], axis=1))
        
        current_r = min(np.min(d_pairs), np.min(d_boundary))
        
        # Find points that are close to the current radius (active)
        active_pair = np.where(d_pairs < current_r * 1.05)[0]
        active_bound = np.where(d_boundary < current_r * 1.05)[0]
        active_idx = np.unique(np.concatenate([active_pair, active_bound]))
        
        if len(active_idx) == 0:
            break
        
        # Optimize only active points
        if len(active_idx) >= 2:
            sub_pts = best_pts[active_idx].copy()
            
            def sub_obj(x):
                sub_p = x.reshape(-1, 2)
                sub_p = np.clip(sub_p, -0.5, 0.5)
                
                # Distances among active points
                d_sub = cdist(sub_p, sub_p)
                np.fill_diagonal(d_sub, np.inf)
                d_sub_pairs = np.min(d_sub, axis=1) / 2.0
                
                # Distances from active to all other points
                other_idx = np.setdiff1d(np.arange(N), active_idx)
                if len(other_idx) > 0:
                    d_cross = cdist(sub_p, best_pts[other_idx])
                    d_cross_pairs = np.min(d_cross, axis=1) / 2.0
                    d_pairs = np.min(np.concatenate([d_sub_pairs, d_cross_pairs]))
                else:
                    d_pairs = np.min(d_sub_pairs)
                
                d_boundary = np.min(np.stack([0.5 - np.abs(sub_p[:, 0]), 0.5 - np.abs(sub_p[:, 1])], axis=1))
                
                all_d = np.concatenate([d_pairs, d_boundary])
                alpha = 1000
                min_val = np.min(all_d)
                exp_sum = np.sum(np.exp(-alpha * (all_d - min_val)))
                soft_min = min_val - np.log(exp_sum) / alpha
                return -soft_min
            
            try:
                res = minimize(
                    sub_obj, sub_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * len(active_idx)),
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if time.time() - start > time_limit:
                    break
                
                best_pts[active_idx] = res.x.reshape(-1, 2)
                r = _compute_radius(best_pts)
                if r > best_r:
                    best_r = r
            except Exception:
                pass
    
    return best_pts, best_r


def construct():
    start = time.time()
    np.random.seed(42)
    
    initializers = [
        _initialize_hexagonal,
        _initialize_jittered_hex,
        _initialize_grid,
        _initialize_random,
    ]
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Try each initializer with multi-phase optimization
    for init_func in initializers:
        if time.time() - start > DEADLINE - 10:
            break
        pts = init_func()
        remaining = DEADLINE - time.time() - 10
        if remaining <= 0:
            break
        pts, r = _local_optimize(pts, remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Phase 2: Additional random restarts if time permits
    restart_count = 0
    while time.time() - start < DEADLINE - 15 and restart_count < 5:
        pts = _initialize_random()
        remaining = DEADLINE - time.time() - 15
        if remaining <= 0:
            break
        pts, r = _local_optimize(pts, remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        restart_count += 1
    
    if best_pts is None:
        best_pts = _initialize_hexagonal()
    
    # Phase 3: High-precision polish
    if time.time() - start < DEADLINE - 5:
        remaining = DEADLINE - time.time() - 5
        best_pts, r = _polish_high_precision(best_pts, remaining)
        if r > best_r:
            best_r = r
    
    # Phase 4: Active set polish
    if time.time() - start < DEADLINE - 2:
        remaining = DEADLINE - time.time() - 2
        best_pts, r = _active_set_polish(best_pts, remaining)
        if r > best_r:
            best_r = r
    
    # Final validation
    final_r = _compute_radius(best_pts)
    if final_r <= 0:
        # Fallback to a safe configuration
        best_pts = _initialize_grid()
    
    return best_pts
