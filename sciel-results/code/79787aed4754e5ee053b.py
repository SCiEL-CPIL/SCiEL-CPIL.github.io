# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _base_inside(u, v):
    """Vectorized base corridor check."""
    horiz = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -3.0 - 1e-9) & (v <= 1.0 + 1e-9)
    vert = (u >= -3.0 - 1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
    return horiz | vert

def _build_grid(subsample=1):
    """Build sofa grid. subsample=1 for full, >1 for subsampled."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    if subsample > 1:
        xs = xs[::subsample]
        ys = ys[::subsample]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return np.column_stack([X.ravel(), Y.ravel()])

def _densify_path(path_rows):
    """Build all placements with densification between consecutive rows."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    placements = [start]
    prev = start
    for row in path_rows:
        a1 = prev[2]
        a2 = row[2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for j in range(1, m + 1):
            t = j / m
            placements.append([
                prev[0] + t * (row[0] - prev[0]),
                prev[1] + t * (row[1] - prev[1]),
                a1 + t * (a2 - a1),
            ])
        prev = row
    # Add end
    a1 = prev[2]
    a2 = end[2]
    m = max(1, int(np.ceil(abs(a2 - a1))))
    for j in range(1, m + 1):
        t = j / m
        placements.append([
            prev[0] + t * (end[0] - prev[0]),
            prev[1] + t * (end[1] - prev[1]),
            a1 + t * (a2 - a1),
        ])
    return placements

def _eval_path_score(path_rows, W_pts, use_active=True):
    """Compute the sofa area for a given set of rows."""
    placements = _densify_path(path_rows)
    N = len(W_pts)
    mask = np.ones(N, dtype=bool)
    for x, y, a in placements:
        a_rad = np.deg2rad(a)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        dx = W_pts[:, 0] - x
        dy = W_pts[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        local = _base_inside(u, v)
        mask &= local
        if not np.any(mask):
            break
    return np.sum(mask) * 0.0001

def _eval_path_score_smooth(path_rows, W_pts):
    """Smooth surrogate: count points with margin > 0.005 inside all placements."""
    placements = _densify_path(path_rows)
    N = len(W_pts)
    margin = 0.005
    for x, y, a in placements:
        a_rad = np.deg2rad(a)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        dx = W_pts[:, 0] - x
        dy = W_pts[:, 1] - y
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        # Check distance to boundary > margin
        # For each rectangle, compute min distance to boundary
        # horiz: 0<=u<=1, -3<=v<=1
        d_h = np.minimum(np.minimum(u, 1.0 - u), np.minimum(v + 3.0, 1.0 - v))
        d_h = np.where((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1), d_h, -np.inf)
        # vert: -3<=u<=1, 0<=v<=1
        d_v = np.minimum(np.minimum(u + 3.0, 1.0 - u), np.minimum(v, 1.0 - v))
        d_v = np.where((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1), d_v, -np.inf)
        d = np.maximum(d_h, d_v)
        if np.sum(d > margin) < 100:  # Early exit if too few points
            return 0.0
    # Count points that are inside with margin in all placements
    # This is expensive; use a simpler approach
    return _eval_path_score(path_rows, W_pts)

def _arc_path_params(c1, c2, c3, c4, c5, c6):
    """
    Generate 89 path rows from 6 parameters.
    x(t) = c1 + c2*t + c3*t^2 + c4*t^3
    y(t) = c5 + c6*t
    where t = a/90, a in [1, 89]
    """
    angles = np.arange(1, 90, 1.0)
    t = angles / 90.0
    x = c1 + c2*t + c3*t**2 + c4*t**3
    y = c5 + c6*t
    rows = np.column_stack([x, y, angles])
    return rows

def _initial_paths():
    """Generate several initial arc parameter sets."""
    paths = []
    # Original arc
    paths.append([0.0, -0.5, 0.0, 0.0, 0.1, 0.2])
    # Slight variations
    paths.append([0.1, -0.6, 0.1, 0.0, 0.0, 0.3])
    paths.append([-0.1, -0.4, 0.2, -0.1, 0.2, 0.1])
    paths.append([0.0, -0.5, 0.1, 0.1, 0.15, 0.15])
    paths.append([-0.05, -0.45, 0.15, -0.05, 0.05, 0.25])
    return paths

def _nelder_mead(func, x0, bounds=None, max_iter=200, tol=1e-8, time_limit=None):
    """Nelder-Mead optimization."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbed = np.array(x0, dtype=float)
        step = 0.1 if bounds is None else (bounds[i][1] - bounds[i][0]) * 0.1
        perturbed[i] += step
        simplex.append(perturbed)
    
    fvals = [func(x) for x in simplex]
    best_idx = np.argmin(fvals)
    best_x = simplex[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time_limit is not None and _elapsed() > time_limit:
            break
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + 1.0 * (centroid - simplex[-1])
        f_ref = func(x_ref)
        
        if f_ref < fvals[0]:
            # Expansion
            x_exp = centroid + 2.0 * (centroid - simplex[-1])
            f_exp = func(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + 0.5 * (simplex[-1] - centroid)
            f_con = func(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        # Track best
        min_idx = np.argmin(fvals)
        if fvals[min_idx] < best_f:
            best_f = fvals[min_idx]
            best_x = simplex[min_idx].copy()
    
    return best_x, best_f

def _refine_path(path_rows, W_pts, time_limit):
    """Refine a path using coordinate descent on individual rows."""
    best_path = path_rows.copy()
    best_score = _eval_path_score(best_path, W_pts)
    step = 0.03
    n_passes = 0
    while _elapsed() < time_limit:
        improved = False
        for k in range(len(best_path)):
            if _elapsed() > time_limit:
                break
            for dx in [step, -step]:
                if _elapsed() > time_limit:
                    break
                trial = best_path.copy()
                trial[k, 0] += dx
                s = _eval_path_score(trial, W_pts)
                if s > best_score + 1e-10:
                    best_path = trial
                    best_score = s
                    improved = True
            for dy in [step, -step]:
                if _elapsed() > time_limit:
                    break
                trial = best_path.copy()
                trial[k, 1] += dy
                s = _eval_path_score(trial, W_pts)
                if s > best_score + 1e-10:
                    best_path = trial
                    best_score = s
                    improved = True
        n_passes += 1
        if not improved:
            step *= 0.7
            if step < 0.005:
                break
        if n_passes % 3 == 0 and step > 0.01:
            step *= 0.7
    return best_path, best_score

def _random_restart_refine(path_rows, W_pts, time_limit, n_restarts=3):
    """Try random perturbations of the path and refine."""
    best_path = path_rows.copy()
    best_score = _eval_path_score(best_path, W_pts)
    for _ in range(n_restarts):
        if _elapsed() > time_limit:
            break
        noise = np.random.normal(0, 0.05, (len(path_rows), 2))
        rp = path_rows.copy()
        rp[:, :2] += noise
        rs = _eval_path_score(rp, W_pts)
        if rs > best_score:
            best_path = rp
            best_score = rs
            # Refine
            best_path, best_score = _refine_path(best_path, W_pts, time_limit)
    return best_path, best_score

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    time_limit = _DEADLINE - 10.0  # Safety margin
    
    # Build full grid
    W_pts = _build_grid(subsample=1)
    N = len(W_pts)
    
    # Generate initial paths from arc parameters
    init_paths = _initial_paths()
    
    best_path = None
    best_score = 0.0
    
    # Optimize each initial path using Nelder-Mead on arc parameters
    for init_params in init_paths:
        if _elapsed() > time_limit:
            break
        
        path_rows = _arc_path_params(*init_params)
        score = _eval_path_score(path_rows, W_pts)
        
        # Nelder-Mead optimization on arc parameters
        def objective(params):
            p = _arc_path_params(*params)
            return -_eval_path_score(p, W_pts)
        
        opt_params, opt_neg_score = _nelder_mead(
            objective, init_params, max_iter=100, tol=1e-6, time_limit=time_limit
        )
        opt_path = _arc_path_params(*opt_params)
        opt_score = -opt_neg_score
        
        if opt_score > best_score:
            best_score = opt_score
            best_path = opt_path
    
    # Refine the best path with coordinate descent
    if best_path is not None:
        best_path, best_score = _refine_path(best_path, W_pts, time_limit)
        
        # Try random restarts
        best_path, best_score = _random_restart_refine(
            best_path, W_pts, time_limit, n_restarts=5
        )
    
    if best_path is None:
        # Fallback: use first initial path
        best_path = _arc_path_params(*init_paths[0])
    
    return best_path.astype(np.float64)


# EVOLVE-BLOCK-END
