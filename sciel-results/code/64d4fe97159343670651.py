"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _greedy_lattice_select(grid_size, stagger=True, time_limit=None):
    """
    Greedy best-fit selection from a fine lattice of candidate sites.
    Select N points maximizing the minimum distance to already-selected neighbors
    and the square boundaries.
    
    Parameters:
    - grid_size: number of divisions per side for the candidate lattice
    - stagger: if True, use a hexagonal/staggered lattice; if False, square lattice
    - time_limit: max seconds for this greedy pass
    
    Returns: (points_array, radius)
    """
    if time_limit is None:
        time_limit = 20.0
    t0 = time.time()
    
    # Generate candidate lattice sites
    if stagger:
        # Hexagonal lattice
        dx = 1.0 / grid_size
        dy = dx * np.sqrt(3) / 2.0
        # Adjust to fit in square
        n_rows = int(1.0 / dy) + 2
        n_cols = grid_size + 2
        candidates = []
        for i in range(n_rows):
            y = -0.5 + i * dy
            if y < -0.5 or y > 0.5:
                continue
            shift = dx / 2.0 if i % 2 == 1 else 0.0
            for j in range(n_cols):
                x = -0.5 + shift + j * dx
                if x >= -0.5 and x <= 0.5:
                    candidates.append((x, y))
        candidates = np.array(candidates)
    else:
        # Square lattice
        dx = 1.0 / grid_size
        dy = 1.0 / grid_size
        xs = np.linspace(-0.5, 0.5, grid_size + 1)
        ys = np.linspace(-0.5, 0.5, grid_size + 1)
        candidates = np.array([(x, y) for y in ys for x in xs])
    
    n_cand = len(candidates)
    if n_cand < N:
        return None, 0.0
    
    # Greedy selection: at each step, pick the candidate that maximizes
    # the minimum distance to already-selected points AND to the walls.
    selected = []
    # Start with corners or center for better initialization
    # Use a simple heuristic: start with the point farthest from all walls
    # Actually, let's just do pure greedy from empty set
    
    # Precompute distance to walls for each candidate
    wall_dist = np.min([
        candidates[:, 0] + 0.5,
        0.5 - candidates[:, 0],
        candidates[:, 1] + 0.5,
        0.5 - candidates[:, 1],
    ], axis=0)
    
    # Track minimum distance to selected points for each candidate
    min_dist_to_selected = np.full(n_cand, np.inf)
    
    for step in range(N):
        if time.time() - t0 > time_limit:
            break
        # Score for each candidate: min(wall_dist, min_dist_to_selected)
        # But we want to maximize the final radius, which is the min over all selected
        # Greedy: pick the candidate with the largest current "bottleneck"
        score = np.minimum(wall_dist, min_dist_to_selected)
        # Pick the best candidate
        idx = np.argmax(score)
        selected.append(idx)
        
        # Update min_dist_to_selected for remaining candidates
        if step < N - 1:
            new_pt = candidates[idx]
            # Compute distances from new_pt to all other candidates
            d = np.sqrt(np.sum((candidates - new_pt) ** 2, axis=1))
            d[idx] = np.inf
            min_dist_to_selected = np.minimum(min_dist_to_selected, d)
    
    if len(selected) < N:
        return None, 0.0
    
    pts = candidates[selected[:N]]
    r = _compute_radius(pts)
    return pts, r


def _rect_grid_score(params, rows, cols):
    """Score (radius) for a rectangular grid. params = [dx, dy, x0, y0].
    We constrain circles to be inside, so radius = min(dx/2, dy/2, dist_to_wall).
    Actually, for a grid, the radius is limited by half the minimum spacing and
    the minimum distance to a wall. Let's compute it directly.
    """
    dx, dy, x0, y0 = params
    if dx <= 0 or dy <= 0:
        return -1e10
    # Generate points
    xs = x0 + np.arange(cols) * dx
    ys = y0 + np.arange(rows) * dy
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    # Check all inside [-0.5, 0.5]^2
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    if N > 1:
        # Use cdist for speed
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
    else:
        min_pair = np.inf
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _hex_grid_score(params, rows, cols):
    """Score for a hexagonal (staggered) grid.
    params = [dx, dy, x0, y0, offset]
    offset: shift of odd rows (0 = no stagger, 0.5*dx = full stagger)
    """
    dx, dy, x0, y0, off = params
    if dx <= 0 or dy <= 0:
        return -1e10
    pts = []
    for i in range(rows):
        y = y0 + i * dy
        shift = off * dx if i % 2 == 1 else 0.0
        xs = x0 + shift + np.arange(cols) * dx
        for x in xs:
            pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _neg_rect_score(params, rows, cols):
    return -_rect_grid_score(params, rows, cols)


def _neg_hex_score(params, rows, cols):
    return -_hex_grid_score(params, rows, cols)


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = _best_result()
    
    # First, try greedy lattice selection with various grid sizes
    for grid_size in [15, 20, 25, 30, 40]:
        if time.time() - t0 > _TIME_LIMIT - 30:  # Leave time for other methods
            break
        for stagger in [True, False]:
            if time.time() - t0 > _TIME_LIMIT - 30:
                break
            try:
                pts, r = _greedy_lattice_select(grid_size, stagger, time_limit=15.0)
                if pts is not None and r > best_r:
                    best_r = r
                    best_pts = pts
            except Exception:
                pass

    # Try rectangular grids with various row/col splits
    # 200 = rows * cols, rows >= cols (tall grids) or we can also try wide
    factor_pairs = []
    for c in range(1, int(np.sqrt(N)) + 1):
        if N % c == 0:
            r = N // c
            factor_pairs.append((r, c))
            factor_pairs.append((c, r))
    # Also add near-factor pairs for flexibility (not required to be exact factors,
    # but we'll only use exact for simplicity)
    factor_pairs = list(set(factor_pairs))

    # Also try non-factor splits by truncating/padding? No, keep exact.
    # But we can also try rows*cols > N and just use N points from the grid.
    # Let's add some additional splits.
    extra = [(14, 15), (15, 14), (20, 10), (10, 20), (25, 8), (8, 25),
             (16, 13), (13, 16), (12, 17), (17, 12), (20, 10), (10, 20)]
    for pair in extra:
        if pair not in factor_pairs:
            factor_pairs.append(pair)

    # Rectangular grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        if rows * cols < N:
            continue
        # Initial guess: uniform grid
        dx0 = 1.0 / cols
        dy0 = 1.0 / rows
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        x0 = np.array([dx0, dy0, x00, y00])
        try:
            res = minimize(_neg_rect_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v = res.x
                xs = x0v + np.arange(cols) * dx
                ys = y0v + np.arange(rows) * dy
                pts = np.array([(x, y) for y in ys for x in xs])[:N]
                if np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Hexagonal grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        if rows * cols < N:
            continue
        dx0 = 1.0 / cols
        dy0 = dx0 * np.sqrt(3) / 2
        # Adjust so it fits: total height = (rows-1)*dy0, need <= 1
        if (rows - 1) * dy0 > 1.0:
            dy0 = 1.0 / (rows - 1) if rows > 1 else 1.0
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        off0 = 0.5  # full stagger
        x0 = np.array([dx0, dy0, x00, y00, off0])
        try:
            res = minimize(_neg_hex_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v, offv = res.x
                pts = []
                for i in range(rows):
                    y = y0v + i * dy
                    shift = offv * dx if i % 2 == 1 else 0.0
                    xs = x0v + shift + np.arange(cols) * dx
                    for x in xs:
                        pts.append((x, y))
                pts = np.array(pts[:N])
                if len(pts) == N and \
                   np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Final polish: try to improve best_pts with local optimization
    if time.time() - t0 < _TIME_LIMIT - 5 and best_r > 0:
        try:
            pts = best_pts.copy()
            
            def _neg_radius_direct(params):
                """Direct radius minimization for local polish."""
                p = params.reshape(-1, 2)
                if np.any(p[:, 0] < -0.5) or np.any(p[:, 0] > 0.5) or \
                   np.any(p[:, 1] < -0.5) or np.any(p[:, 1] > 0.5):
                    return 1e10
                wall = np.min([
                    np.min(p[:, 0]) + 0.5,
                    0.5 - np.max(p[:, 0]),
                    np.min(p[:, 1]) + 0.5,
                    0.5 - np.max(p[:, 1]),
                ])
                from scipy.spatial.distance import cdist
                D = cdist(p, p)
                np.fill_diagonal(D, np.inf)
                min_pair = np.min(D)
                r = min(wall, min_pair / 2.0)
                return -r if r > 0 else 1e10
            
            # Only do local polish if we have time
            if time.time() - t0 < _TIME_LIMIT - 10:
                x0 = pts.flatten()
                res = minimize(_neg_radius_direct, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                r = -res.fun
                if r > best_r:
                    new_pts = res.x.reshape(-1, 2)
                    if np.all(new_pts[:, 0] >= -0.5) and np.all(new_pts[:, 0] <= 0.5) and \
                       np.all(new_pts[:, 1] >= -0.5) and np.all(new_pts[:, 1] <= 0.5):
                        best_r = r
                        best_pts = new_pts
        except Exception:
            pass
    
    if best_r <= 0:
        return _best_result()
    return best_pts
