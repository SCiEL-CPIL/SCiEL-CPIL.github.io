"""Circle packing in unit square using L-BFGS-B with barrier penalty.
Formulates as continuous optimization: minimize -sum(r) + penalty for overlaps/boundary.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0
START = time.time()


def _elapsed():
    return time.time() - START


def _unpack(v):
    """Unpack flat vector into x, y, r."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    return x, y, r


def _pack(x, y, r):
    """Pack x, y, r into flat vector."""
    return np.concatenate([x, y, r])


def _penalized(v, penalty_weight=100.0):
    """Objective: -sum(r) + penalty for overlaps and boundary violations."""
    x, y, r = _unpack(v)
    
    # Boundary penalty: circle must be inside [0,1]^2
    boundary_viol = np.maximum(0, r - np.minimum(x, 1.0 - x)) + \
                    np.maximum(0, r - np.minimum(y, 1.0 - y))
    
    # Overlap penalty: for each pair, penalize if dist < r_i + r_j
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            need = r[i] + r[j]
            if dist < need:
                overlap_pen += (need - dist) ** 2
    
    total_pen = boundary_viol.sum() + overlap_pen
    return -np.sum(r) + penalty_weight * total_pen


def _gradient(v, penalty_weight=100.0, eps=1e-7):
    """Numerical gradient of penalized objective."""
    grad = np.zeros_like(v)
    for i in range(len(v)):
        v_plus = v.copy()
        v_plus[i] += eps
        v_minus = v.copy()
        v_minus[i] -= eps
        grad[i] = (_penalized(v_plus, penalty_weight) - _penalized(v_minus, penalty_weight)) / (2 * eps)
    return grad


def _is_valid(x, y, r, tol=1e-6):
    """Check feasibility: no overlaps, all inside square."""
    # Boundary check
    if np.any(r + tol > np.minimum(x, 1.0 - x)) or np.any(r + tol > np.minimum(y, 1.0 - y)):
        return False
    
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _grid_init(rng, r0=0.18):
    """Create initial grid configuration."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    pts.append((0.5, 0.5 + 0.5 / cols))
    rng.shuffle(pts)
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, r0)
    return x, y, r


def _fix_to_boundary(x, y, r):
    """Ensure circles fit within the square."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.minimum(r, np.maximum(d, 0.0))


def _optimize_single(v0, penalty_weight=100.0, max_iter=500):
    """Run L-BFGS-B optimization from initial state v0."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.45)] * N
    
    result = minimize(
        _penalized,
        v0,
        args=(penalty_weight,),
        method='L-BFGS-B',
        jac=_gradient,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return result.x


def _construct_optimized():
    """Main optimization: try multiple initial configs, return best valid packing."""
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)
    
    # Try several initial configurations with different radii
    r0_values = [0.16, 0.18, 0.20, 0.15]
    
    for r0 in r0_values:
        if _elapsed() > DEADLINE - 15.0:
            break
        
        x0, y0, r0_arr = _grid_init(rng, r0)
        r0_arr = _fix_to_boundary(x0, y0, r0_arr)
        v0 = _pack(x0, y0, r0_arr)
        
        # Run optimization
        v_opt = _optimize_single(v0, penalty_weight=100.0, max_iter=300)
        x, y, r = _unpack(v_opt)
        
        # Post-process: clip to boundary
        r = _fix_to_boundary(x, y, r)
        
        # Check validity
        if _is_valid(x, y, r):
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best = np.column_stack([x, y, r])
    
    return best, best_val


def construct():
    best, best_val = _construct_optimized()
    
    if best is not None:
        return best
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
