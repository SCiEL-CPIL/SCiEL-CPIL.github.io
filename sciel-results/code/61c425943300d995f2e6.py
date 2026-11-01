"""Packing 200 equal circles in a unit square: local search from diverse starts."""
import numpy as np
import time

N = 200
DEADLINE = 140.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary: min over each point of min(0.5-x, x+0.5, 0.5-y, y+0.5)
    # i.e., 0.5 - max(|x|, |y|) per point... actually it's 0.5 - |x| and 0.5 - |y|
    # The constraint is 0.5 - |x| >= r and 0.5 - |y| >= r, so boundary term = min_i min(0.5-|xi|, 0.5-|yi|)
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    # Use scipy cdist if available, else manual
    from scipy.spatial.distance import cdist
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _fast_score(pts):
    """Faster scoring using vectorized pairwise distances."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()


def _hex_init():
    # Hexagonal lattice packing
    # For N points, find rows and columns
    # Hexagonal: row spacing = s * sqrt(3)/2, col spacing = s
    # We want to fit as many as possible in [-0.5, 0.5]^2
    # Start with a reasonable spacing and adjust
    best_pts = None
    best_r = 0
    for s in np.linspace(0.03, 0.05, 20):
        # Generate hex lattice points
        pts_list = []
        row = 0
        while True:
            y = -0.5 + 0.5 * s + row * s * np.sqrt(3) / 2
            if y > 0.5 - 0.5 * s:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + 0.5 * s + offset
            while x <= 0.5 - 0.5 * s:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _fast_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        return _grid_init()
    return best_pts


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    return pts


def _local_search(pts, max_iters=50000, seed=0, time_budget=10.0):
    """Local search: perturb random points, accept if score improves."""
    rng = np.random.default_rng(seed)
    pts = pts.copy()
    best_r = _fast_score(pts)
    best_pts = pts.copy()
    
    # Start with a moderate step size
    step_size = 0.02
    no_improve = 0
    t0 = time.time()
    
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Pick a random point to perturb
        idx = rng.integers(0, N)
        old_pt = pts[idx].copy()
        
        # Gaussian perturbation
        pts[idx] = old_pt + rng.normal(0, step_size, size=2)
        
        # Clamp to square
        pts[idx, 0] = np.clip(pts[idx, 0], -0.5, 0.5)
        pts[idx, 1] = np.clip(pts[idx, 1], -0.5, 0.5)
        
        new_r = _fast_score(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
            no_improve = 0
            # Occasionally increase step size slightly
            step_size = min(step_size * 1.01, 0.05)
        else:
            pts[idx] = old_pt
            no_improve += 1
            if no_improve > 200:
                step_size *= 0.95
                no_improve = 0
                if step_size < 0.001:
                    break
    
    return best_pts, best_r


def _simulated_annealing(pts, max_iters=30000, seed=0, time_budget=10.0):
    """Simulated annealing for circle packing."""
    rng = np.random.default_rng(seed)
    pts = pts.copy()
    best_r = _fast_score(pts)
    best_pts = pts.copy()
    
    T0 = 0.01
    T_end = 0.0001
    t0 = time.time()
    
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        T = T0 * (T_end / T0) ** (it / max_iters)
        
        idx = rng.integers(0, N)
        old_pt = pts[idx].copy()
        pts[idx] = old_pt + rng.normal(0, 0.01, size=2)
        pts[idx, 0] = np.clip(pts[idx, 0], -0.5, 0.5)
        pts[idx, 1] = np.clip(pts[idx, 1], -0.5, 0.5)
        
        new_r = _fast_score(pts)
        delta = new_r - best_r
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            if new_r > best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            pts[idx] = old_pt
    
    return best_pts, best_r


def construct():
    t_start = time.time()
    
    # Try multiple initializations and local searches
    initializers = [
        ("grid", _grid_init),
        ("hex", _hex_init),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(1)),
        ("rand2", lambda: _random_init(42)),
        ("rand3", lambda: _random_init(123)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_fn in initializers:
        if time.time() > DEADLINE - 5:
            break
        
        pts = init_fn()
        r = _fast_score(pts)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Local search from this start
        remaining = DEADLINE - time.time()
        if remaining < 2:
            break
        
        time_budget = min(remaining * 0.15, 12.0)
        ls_pts, ls_r = _local_search(pts, max_iters=30000, seed=hash(name) % 10000, time_budget=time_budget)
        
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_pts.copy()
    
    # If we have time, try simulated annealing from the best
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        time_budget = min(remaining * 0.4, 20.0)
        sa_pts, sa_r = _simulated_annealing(best_pts, max_iters=20000, seed=7, time_budget=time_budget)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
    
    # Final local search polish
    remaining = DEADLINE - time.time()
    if remaining > 3 and best_pts is not None:
        time_budget = min(remaining * 0.5, 15.0)
        ls_pts, ls_r = _local_search(best_pts, max_iters=50000, seed=99, time_budget=time_budget)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_pts.copy()
    
    if best_pts is None:
        best_pts = _grid_init()
    
    # Ensure finite and valid
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
