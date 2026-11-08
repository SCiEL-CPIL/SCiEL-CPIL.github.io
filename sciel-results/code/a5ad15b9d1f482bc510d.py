"""Circle packing in perimeter-4 rectangle: Voronoi radii + SA exploration + L-BFGS-B polish."""
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _voronoi_inradii(cx, cy, w, h):
    """
    Compute the inradius of each Voronoi cell clipped to the box [0,w]x[0,h].
    Returns array of N radii.
    """
    n = len(cx)
    pts = np.column_stack([cx, cy])
    
    # Add boundary points to constrain Voronoi diagram
    # We need points outside the box to clip cells properly
    margin = max(w, h) * 0.5
    boundary_pts = np.array([
        [-margin, -margin], [w + margin, -margin],
        [-margin, h + margin], [w + margin, h + margin],
        [-margin, h/2], [w + margin, h/2],
        [w/2, -margin], [w/2, h + margin],
    ])
    
    all_pts = np.vstack([pts, boundary_pts])
    
    try:
        vor = Voronoi(all_pts)
    except Exception:
        # Fallback: use simple min-distance approach
        bx = np.minimum(cx, w - cx)
        by = np.minimum(cy, h - cy)
        bdist = np.minimum(bx, by)
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        pair_min = np.min(dist, axis=1) / 2.0
        return np.maximum(np.minimum(bdist, pair_min), 1e-8)
    
    radii = np.zeros(n)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded region (shouldn't happen with boundary points)
            # Fallback to boundary distance
            bx = min(cx[i], w - cx[i])
            by = min(cy[i], h - cy[i])
            radii[i] = max(min(bx, by), 1e-8)
            continue
        
        vertices = vor.vertices[region]
        if len(vertices) == 0:
            radii[i] = 1e-8
            continue
        
        # Compute inradius: minimum distance from center to any edge of the cell
        # But we also need to clip to the box
        # The Voronoi vertices are already clipped by boundary points
        # Compute distance to each edge
        min_dist = np.inf
        for j in range(len(vertices)):
            v1 = vertices[j]
            v2 = vertices[(j + 1) % len(vertices)]
            # Distance from point to line segment
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length_sq = dx*dx + dy*dy
            if length_sq < 1e-15:
                continue
            t = np.clip(((cx[i] - v1[0])*dx + (cy[i] - v1[1])*dy) / length_sq, 0, 1)
            proj_x = v1[0] + t * dx
            proj_y = v1[1] + t * dy
            dist = np.sqrt((cx[i] - proj_x)**2 + (cy[i] - proj_y)**2)
            min_dist = min(min_dist, dist)
        
        # Also clip to box boundaries
        box_dist = min(cx[i], w - cx[i], cy[i], h - cy[i])
        radii[i] = max(min(min_dist, box_dist), 1e-8)
    
    return radii


def _objective(params):
    """
    params: [cx_0, cy_0, ..., cx_{n-1}, cy_{n-1}, w]
    h = 2 - w
    Returns negative sum of Voronoi inradii (to minimize).
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    w = params[2*n]
    h = 2.0 - w

    if w <= 0.1 or h <= 0.1:
        return 1e6

    # Penalty for centers outside box
    ox = np.maximum(0.0, -cx, cx - w)
    oy = np.maximum(0.0, -cy, cy - h)
    pen = np.sum(ox**2 + oy**2) * 100.0

    r = _voronoi_inradii(cx, cy, w, h)
    return -(np.sum(r)) + pen


def _lbfgsb_optimize(x0, max_iter=200, time_limit=30.0):
    """Run L-BFGS-B with time guard."""
    t0 = time.time()
    
    def _obj(p):
        if time.time() - t0 > time_limit:
            raise TimeoutError()
        return _objective(p)
    
    bounds = []
    n = N
    for i in range(n):
        bounds.append((0.0, 1.9))
    for i in range(n):
        bounds.append((0.0, 1.9))
    bounds.append((0.1, 1.9))
    
    try:
        res = minimize(
            _obj, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-8,
                'gtol': 1e-6,
            }
        )
        return res.x, res.fun
    except TimeoutError:
        return x0, _objective(x0)
    except Exception:
        return x0, _objective(x0)


def _simulated_annealing(x0, rng, n_iter=500, T_start=0.05, T_end=0.001, time_limit=40.0):
    """Simulated annealing to explore configuration space."""
    t0 = time.time()
    current = x0.copy()
    current_val = _objective(current)
    best = current.copy()
    best_val = current_val
    
    n = N
    for i in range(n_iter):
        if time.time() - t0 > time_limit:
            break
        
        T = T_start * (T_end / T_start) ** (i / max(n_iter - 1, 1))
        
        # Propose a move: perturb one or two coordinates
        n_moves = rng.integers(1, 4)
        proposal = current.copy()
        for _ in range(n_moves):
            idx = rng.integers(0, 2 * N + 1)
            step = rng.normal(0, T * 2)
            proposal[idx] += step
        
        # Clamp to valid range
        for i in range(N):
            proposal[i] = np.clip(proposal[i], 0.01, 1.9)
            proposal[N + i] = np.clip(proposal[N + i], 0.01, 1.9)
        proposal[2*N] = np.clip(proposal[2*N], 0.1, 1.9)
        
        # Ensure h > 0.1
        h_prop = 2.0 - proposal[2*N]
        if h_prop < 0.1:
            proposal[2*N] = 1.9
        
        proposal_val = _objective(proposal)
        
        # Accept if better or with probability exp(-delta/T)
        delta = proposal_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = proposal
            current_val = proposal_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
    
    return best, best_val


def _make_grid_init(w, h, rng):
    """Create a grid-like initialization."""
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros(2 * N + 1)
    params[2*N] = w
    for i in range(N):
        ix = i % side
        iy = i // side
        params[i] = (ix + 0.5) / side * w
        params[N + i] = (iy + 0.5) / side * h
    # Add small random perturbation
    params[:N] += rng.uniform(-0.02, 0.02, N)
    params[N:2*N] += rng.uniform(-0.02, 0.02, N)
    # Clamp
    params[:N] = np.clip(params[:N], 0.01, w - 0.01)
    params[N:2*N] = np.clip(params[N:2*N], 0.01, h - 0.01)
    return params


def construct():
    start = time.time()
    rng = np.random.default_rng(42)
    best_params = None
    best_val = np.inf

    # Generate diverse initializations
    inits = []
    
    # Grid in square
    inits.append(_make_grid_init(1.0, 1.0, rng))
    
    # Grid in various aspect ratios
    for w_ratio in [0.5, 0.7, 1.3, 1.5, 1.8]:
        w_try = min(w_ratio, 1.9)
        h_try = 2.0 - w_try
        if h_try > 0.1:
            inits.append(_make_grid_init(w_try, h_try, rng))
    
    # Random inits
    for _ in range(6):
        w_try = rng.uniform(0.4, 1.6)
        h_try = 2.0 - w_try
        params = np.zeros(2 * N + 1)
        params[2*N] = w_try
        params[:N] = rng.uniform(0.02, w_try - 0.02, N)
        params[N:2*N] = rng.uniform(0.02, h_try - 0.02, N)
        inits.append(params)
    
    # Phase 1: Simulated annealing from each init to find good basins
    for idx, x0 in enumerate(inits):
        if time.time() > DEADLINE - 20:
            break
        sa_params, sa_val = _simulated_annealing(x0, rng, n_iter=300, T_start=0.05, T_end=0.001, time_limit=25.0)
        if sa_val < best_val:
            best_val = sa_val
            best_params = sa_params.copy()
    
    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Phase 2: L-BFGS-B polish from best SA result
    if time.time() < DEADLINE - 10:
        polished, pol_val = _lbfgsb_optimize(best_params, max_iter=100, time_limit=20.0)
        if pol_val < best_val:
            best_val = pol_val
            best_params = polished.copy()

    cx = best_params[:N]
    cy = best_params[N:2*N]
    w = best_params[2*N]
    h = 2.0 - w

    # Compute final radii using Voronoi
    r = _voronoi_inradii(cx, cy, w, h)

    # Slightly shrink radii to ensure disjoint interiors (strict inequality)
    r = r * (1.0 - 1e-6)

    result = np.column_stack([cx, cy, r])
    return result
