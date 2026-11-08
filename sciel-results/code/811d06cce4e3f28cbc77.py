"""Circle packing in perimeter-4 rectangle: soft-min NM + SA exploration + L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 138  # safety margin before 150s kill


def _softmin_radii(cx, cy, w, h, k):
    """
    For given centers and box (w,h) with w+h=2:
      r_i = min( min_j dist(i,j)/2 for j!=i, cx_i, cy_i, w-cx_i, h-cy_i )
    Use soft-min approximation for smoothness.
    """
    n = len(cx)
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)

    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0

    r = -k * np.logaddexp(-bdist / k, -pair_min / k)
    return np.maximum(r, 0.0)


def _objective(params, k):
    """
    params: [cx_0, cy_0, ..., cx_{n-1}, cy_{n-1}, w]
    h = 2 - w
    Returns negative sum of radii (to minimize).
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    w = params[2*n]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return 1e6

    # Penalty for centers outside box
    ox = np.maximum(0.0, -cx, cx - w)
    oy = np.maximum(0.0, -cy, cy - h)
    pen = np.sum(ox**2 + oy**2) * 50.0

    r = _softmin_radii(cx, cy, w, h, k)
    return -(np.sum(r)) + pen


def _exact_radii(cx, cy, w, h):
    """Compute exact hard-min radii."""
    n = len(cx)
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0
    r = np.minimum(bdist, pair_min)
    return np.maximum(r, 1e-6)


def _nelder_mead_optimize(x0, k, max_iter=3000, timeout_frac=0.8):
    """Run Nelder-Mead with time guard."""
    t0 = time.time()
    max_time = timeout_frac * 120.0
    
    def _obj(p):
        return _objective(p, k)
    
    try:
        res = minimize(
            _obj, x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True,
            }
        )
        return res.x, res.fun
    except Exception:
        return x0, _obj(x0)


def _lbfgsb_optimize(x0, k, max_iter=5000, timeout_frac=0.8):
    """Run L-BFGS-B with time guard."""
    t0 = time.time()
    max_time = timeout_frac * 120.0
    
    def _obj(p):
        return _objective(p, k)
    
    try:
        res = minimize(
            _obj, x0,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-10,
                'gtol': 1e-6,
                'maxls': 20,
            }
        )
        return res.x, res.fun
    except Exception:
        return x0, _obj(x0)


def _simulated_annealing(x0, k, rng, n_iter=2000, t_start=0.5, t_end=0.01):
    """Simulated annealing to explore configuration space."""
    t0 = time.time()
    max_time = 15.0  # seconds
    
    best_x = x0.copy()
    best_val = _objective(x0, k)
    current_x = x0.copy()
    current_val = best_val
    
    for i in range(n_iter):
        if time.time() > t0 + max_time:
            break
        
        # Adaptive temperature
        frac = i / max(n_iter - 1, 1)
        temp = t_start * (t_end / t_start) ** frac
        
        # Propose a move: perturb one or two coordinates
        n_coords = len(current_x)
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(n_coords, size=n_perturb, replace=False)
        step_size = temp * 0.1
        new_x = current_x.copy()
        new_x[idx] += rng.normal(0, step_size, size=n_perturb)
        
        # Keep w in valid range
        new_x[2*N] = np.clip(new_x[2*N], 0.1, 1.9)
        
        new_val = _objective(new_x, k)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_x = new_x
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
    
    return best_x, best_val


def _polish(cx, cy, w, h, k=0.005, max_iter=500):
    """Final polish pass with tighter tolerance using L-BFGS-B."""
    def _obj(p):
        cx_l = p[:N]
        cy_l = p[N:2*N]
        w_l = p[2*N]
        h_l = 2.0 - w_l
        if w_l <= 0.05 or h_l <= 0.05:
            return 1e6
        ox = np.maximum(0.0, -cx_l, cx_l - w_l)
        oy = np.maximum(0.0, -cy_l, cy_l - h_l)
        pen = np.sum(ox**2 + oy**2) * 50.0
        r = _softmin_radii(cx_l, cy_l, w_l, h_l, k)
        return -(np.sum(r)) + pen
    
    x0 = np.concatenate([cx, cy, [w]])
    try:
        res = minimize(
            _obj, x0,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 30,
            }
        )
        cx_l = res.x[:N]
        cy_l = res.x[N:2*N]
        w_l = res.x[2*N]
        h_l = 2.0 - w_l
        return cx_l, cy_l, w_l, h_l
    except Exception:
        return cx, cy, w, h


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

    # Soft-min parameter: start moderate, tighten later
    k_initial = 0.02
    k_final = 0.005

    # Generate diverse initializations
    inits = []
    
    # Grid in square
    inits.append(_make_grid_init(1.0, 1.0, rng))
    
    # Grid in various aspect ratios
    for w_ratio in [0.5, 0.7, 1.3, 1.5, 2.0]:
        w_try = min(w_ratio, 1.95)
        h_try = 2.0 - w_try
        if h_try > 0.05:
            inits.append(_make_grid_init(w_try, h_try, rng))
    
    # Random inits
    for _ in range(8):
        w_try = rng.uniform(0.4, 1.6)
        h_try = 2.0 - w_try
        params = np.zeros(2 * N + 1)
        params[2*N] = w_try
        params[:N] = rng.uniform(0.02, w_try - 0.02, N)
        params[N:2*N] = rng.uniform(0.02, h_try - 0.02, N)
        inits.append(params)
    
    # Phase 1: Run L-BFGS-B from each init (faster than NM for smooth objectives)
    for idx, x0 in enumerate(inits):
        if time.time() > DEADLINE - 30:
            break
        x_opt, val = _lbfgsb_optimize(x0, k_initial, max_iter=3000, timeout_frac=0.7)
        if val < best_val:
            best_val = val
            best_params = x_opt.copy()
    
    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Phase 2: Simulated annealing to explore and potentially find better basins
    if time.time() < DEADLINE - 25:
        sa_x, sa_val = _simulated_annealing(best_params, k_initial, rng, n_iter=3000, t_start=0.3, t_end=0.005)
        if sa_val < best_val:
            best_val = sa_val
            best_params = sa_x.copy()

    # Phase 3: Refine best candidate with L-BFGS-B
    if time.time() < DEADLINE - 15:
        x_opt, val = _lbfgsb_optimize(best_params, k_final, max_iter=5000, timeout_frac=0.8)
        if val < best_val:
            best_val = val
            best_params = x_opt.copy()

    cx = best_params[:N]
    cy = best_params[N:2*N]
    w = best_params[2*N]
    h = 2.0 - w

    # Final high-precision polish
    if time.time() < DEADLINE - 5:
        cx, cy, w, h = _polish(cx, cy, w, h, k=k_final, max_iter=2000)

    # Compute final radii (hard min, no soft)
    r = _exact_radii(cx, cy, w, h)

    # Slightly shrink radii to ensure disjoint interiors (strict inequality)
    r = r * (1.0 - 1e-6)

    result = np.column_stack([cx, cy, r])
    return result
