"""Circle packing in unit square using direct (x,y,r) optimization.

Key structural change: optimize all 78 variables (positions AND radii) jointly
with a smooth, differentiable objective that directly maximizes sum of radii
while penalizing overlaps and boundary violations. This avoids the
non-differentiable 'compute radii from geometry' step and lets the optimizer
find configurations where radii are not simply half the nearest-neighbor distance.

Uses:
- Direct 78-dim optimization with L-BFGS-B
- Multiple diverse initializations
- Local refinement with SLSQP for tight packing
- Deadline guard for safety
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
DEADLINE = 140.0  # Leave 10s safety margin
START = time.time()


def _elapsed():
    return time.time() - START


def _objective(params):
    """
    Objective to minimize: -sum(r) + overlap_penalty + boundary_penalty.
    params: 78-dim vector [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Clip to valid ranges
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    r = np.clip(r, 1e-6, 0.5)
    
    # Boundary violation: circle must be inside [0,1]^2
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    bound_viol = np.maximum(r - d_bound, 0.0)
    bound_penalty = 100.0 * np.sum(bound_viol ** 2)
    
    # Overlap violation: circles must not overlap
    dist_mat = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    r_sum = r[:, None] + r[None, :]
    overlap_viol = np.maximum(r_sum - dist_mat, 0.0)
    overlap_penalty = 50.0 * np.sum(overlap_viol ** 2)
    
    return -np.sum(r) + overlap_penalty + bound_penalty


def _valid(x, y, r, eps=1e-5):
    """Check if configuration is valid."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_valid(x, y, r, max_iters=100):
    """
    Iteratively shrink radii and adjust positions to eliminate overlaps.
    Returns (x, y, r) that should be valid.
    """
    for _ in range(max_iters):
        # Check boundary
        d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, d_bound)
        
        # Check pairwise overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < r[i] + r[j]:
                    # Shrink both proportionally to fit
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        scale = dist[i, j] / total_r * 0.999
                        r[i] *= scale
                        r[j] *= scale
                    else:
                        r[i] = r[j] = 1e-6
                    changed = True
        
        if not changed:
            break
    
    return x, y, np.maximum(r, 1e-6)


def _grid_init_5x5_plus1(rng):
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    # Initial radii: half the minimum spacing
    spacing = 0.2
    r = np.full(N, spacing * 0.45)
    return x, y, r


def _grid_init_4x7(rng):
    cols, rows = 4, 7
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.12)
    return x, y, r


def _grid_init_6x5(rng):
    cols, rows = 6, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.11)
    return x, y, r


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = rng.uniform(0.05, 0.15, size=N)
    return x, y, r


def _hex_init(rng):
    """Hexagonal lattice initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.19
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.15 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.15 + row * spacing * 0.866
        if x[i] > 0.9:
            x[i] = 0.9
        if y[i] > 0.9:
            y[i] = 0.9
    r = np.full(N, spacing * 0.45)
    return x, y, r


def _optimize_from(x, y, r, method='L-BFGS-B'):
    """Optimize from given initial configuration."""
    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    if method == 'L-BFGS-B':
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6,
                                    'maxfun': 3000})
        except Exception:
            return x, y, r
    elif method == 'SLSQP':
        try:
            res = minimize(_objective, params, method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-8})
        except Exception:
            return x, y, r
    else:
        return x, y, r
    
    v = res.x
    xx = np.clip(v[0::3], 0.0, 1.0)
    yy = np.clip(v[1::3], 0.0, 1.0)
    rr = np.clip(v[2::3], 1e-6, 0.5)
    
    return xx, yy, rr


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x7,
        _grid_init_6x5,
        _random_init,
        _hex_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        x, y, r = init_func(rng)
        
        for restart in range(4):
            if _elapsed() > DEADLINE - 10.0:
                break
            
            # Add small jitter
            x0 = np.clip(x + rng.normal(0, 0.01, N), 0.01, 0.99)
            y0 = np.clip(y + rng.normal(0, 0.01, N), 0.01, 0.99)
            r0 = np.clip(r + rng.normal(0, 0.005, N), 0.01, 0.4)
            
            # First pass: L-BFGS-B
            xx, yy, rr = _optimize_from(x0, y0, r0, method='L-BFGS-B')
            
            # Fix to valid
            xx, yy, rr = _fix_to_valid(xx, yy, rr)
            
            # Second pass: SLSQP refinement
            if _elapsed() < DEADLINE - 15.0:
                xx2, yy2, rr2 = _optimize_from(xx, yy, rr, method='SLSQP')
                xx2, yy2, rr2 = _fix_to_valid(xx2, yy2, rr2)
                
                if _valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])
            
            # Check first pass result
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
