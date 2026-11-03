import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin under 150s kill


def _elapsed(start):
    return time.time() - start


def _penalized(params, w_overlap=100.0, w_bound=100.0):
    """
    Vectorized penalty-based objective.
    Minimize: -sum(r) + w_bound * boundary_penalty + w_overlap * overlap_penalty
    params: shape (N, 3) = [x, y, r]
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]

    obj = -np.sum(r)

    # Boundary penalty
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    over = r - d
    pen_bound = w_bound * np.sum(np.maximum(over, 0.0) ** 2)

    # Overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    viol = need - dist
    pen_overlap = w_overlap * np.sum(np.maximum(viol, 0.0) ** 2)

    return obj + pen_bound + pen_overlap


def _valid(x, y, r, eps=1e-6):
    """Check if configuration is valid (no overlaps, inside square)."""
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


def _fix_to_boundary(x, y, r):
    """Clip radii so each circle fits inside the square."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.minimum(r, np.maximum(d, 0.0))


def _grid_init_5x5_plus1(rng):
    """5x5 grid with one extra circle placed in a gap."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    # Add one extra near center-top
    extra = [(0.5, 0.1)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.18)
    return x, y, r


def _grid_init_4x6_plus2(rng):
    """4x6 grid (24 circles) plus 2 extra."""
    cols, rows = 4, 6
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    # Add 2 extra
    extra = [(0.5, 0.5), (0.25, 0.75)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.19)
    return x, y, r


def _grid_init_square(rng):
    """Simple square grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    return x, y, r


def construct():
    start_time = time.time()
    
    best_params = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    # Multiple initialization strategies
    init_strategies = [
        _grid_init_square,
        _grid_init_5x5_plus1,
        _grid_init_4x6_plus2,
    ]

    # Try multiple initializations with multi-start
    for init_func in init_strategies:
        if _elapsed(start_time) > DEADLINE - 5.0:
            break
            
        x, y, r = init_func(rng)
        r = _fix_to_boundary(x, y, r)
        
        # Multi-start with jitter
        for restart in range(4):
            if _elapsed(start_time) > DEADLINE - 5.0:
                break
                
            x0 = x.copy()
            y0 = y.copy()
            r0 = r.copy()
            
            # Add small random noise
            noise_scale = 0.05
            x0 += rng.uniform(-noise_scale, noise_scale, N)
            y0 += rng.uniform(-noise_scale, noise_scale, N)
            r0 += rng.uniform(-noise_scale, noise_scale, N)
            
            # Clip to reasonable bounds
            x0 = np.clip(x0, 0.01, 0.99)
            y0 = np.clip(y0, 0.01, 0.99)
            r0 = np.clip(r0, 0.01, 0.4)
            
            params0 = np.column_stack([x0, y0, r0])
            
            # Optimize using L-BFGS-B
            try:
                res = minimize(
                    _penalized,
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-6}
                )
                
                if res.success:
                    candidate = res.x
                    xx = candidate[:, 0]
                    yy = candidate[:, 1]
                    rr = candidate[:, 2]
                    
                    # Ensure radii are positive
                    rr = np.maximum(rr, 1e-6)
                    
                    # Fix boundary violations
                    rr = _fix_to_boundary(xx, yy, rr)
                    
                    # Check validity
                    if _valid(xx, yy, rr):
                        val = float(np.sum(rr))
                        if val > best_val:
                            best_val = val
                            best_params = np.column_stack([xx, yy, rr])
            except Exception:
                pass

    # If no valid solution found, use a trivial grid
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        cells = cells[:N]
        while len(cells) < N:
            cells.append((0.5 / side, 0.5 / side, 0.1 / side))
        best_params = np.array(cells[:N])
    
    # Final cleanup: ensure strict validity by shrinking slightly if needed
    final_params = best_params.copy()
    for _ in range(10):
        changed = False
        # Fix boundaries
        for i in range(N):
            min_dist = min(final_params[i, 0], 1 - final_params[i, 0], 
                          final_params[i, 1], 1 - final_params[i, 1])
            if final_params[i, 2] > min_dist:
                final_params[i, 2] = min_dist
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_params[i, 0] - final_params[j, 0]
                dy = final_params[i, 1] - final_params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = final_params[i, 2] + final_params[j, 2]
                if d < min_dist:
                    # Shrink the smaller radius to resolve overlap
                    if final_params[i, 2] <= final_params[j, 2]:
                        new_r_i = d - final_params[j, 2]
                        if new_r_i > 0:
                            final_params[i, 2] = new_r_i
                        else:
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
                    else:
                        new_r_j = d - final_params[i, 2]
                        if new_r_j > 0:
                            final_params[j, 2] = new_r_j
                        else:
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
        if not changed:
            break
            
    return final_params
