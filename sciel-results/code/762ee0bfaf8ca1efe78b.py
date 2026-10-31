"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP with diverse initializations."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 138.0  # Safety margin before 150s kill

def _rect_from_width(W):
    H = 2.0 - W
    return W, H

def _objective_penalty(x, W, H, mu=1e3):
    """Return (value, grad) where value = -sum(r) + mu * penalty.
    x is shape (N,3): cx, cy, r.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    # Clamp to a reasonable range for numerical stability
    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    # Objective: maximize sum(r) -> minimize -sum(r)
    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary: r <= min(cx, cy, W-cx, H-cy), r > 0
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    # violation_b[i] = max(r[i] - lo[i], 0)
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)

    # grad for boundary violation
    for i in range(n):
        if viol_b[i] <= 0:
            continue
        # determine which boundary is active (smallest)
        d_left = cx[i]
        d_bottom = cy[i]
        d_right = W - cx[i]
        d_top = H - cy[i]
        min_d = lo[i]
        # For numerical robustness, if multiple are equal, average
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        if d_left <= min_d + eps:
            g_cx += 1.0
        if d_right <= min_d + eps:
            g_cx += -1.0
        if d_bottom <= min_d + eps:
            g_cy += 1.0
        if d_top <= min_d + eps:
            g_cy += -1.0
        # normalize if multiple active
        cnt = (d_left <= min_d + eps) + (d_right <= min_d + eps) + \
              (d_bottom <= min_d + eps) + (d_top <= min_d + eps)
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
        # d viol_b / d cx = -d lo / d cx = -g_cx (since viol = r - lo, r doesn't depend on cx)
        # d viol_b / d r = 1
        dv = 2.0 * mu * viol_b[i]
        grad[i, 0] += dv * (-g_cx)
        grad[i, 1] += dv * (-g_cy)
        grad[i, 2] += dv * 1.0

    # r > 0: violation if r < 0
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    for i in range(n):
        if viol_r[i] > 0:
            dv = 2.0 * mu * viol_r[i]
            grad[i, 2] += dv * (-1.0)

    # Centers in box (soft)
    mu_c = mu * 0.1
    viol_cx_lo = np.maximum(-cx, 0.0)
    viol_cx_hi = np.maximum(cx - W, 0.0)
    viol_cy_lo = np.maximum(-cy, 0.0)
    viol_cy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(viol_cx_lo**2) + np.sum(viol_cx_hi**2) +
                   np.sum(viol_cy_lo**2) + np.sum(viol_cy_hi**2))
    for i in range(n):
        if viol_cx_lo[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_lo[i] * (-1.0)
        if viol_cx_hi[i] > 0:
            grad[i, 0] += 2.0 * mu_c * viol_cx_hi[i] * 1.0
        if viol_cy_lo[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_lo[i] * (-1.0)
        if viol_cy_hi[i] > 0:
            grad[i, 1] += 2.0 * mu_c * viol_cy_hi[i] * 1.0

    # Non-overlap: for each pair, max(r_i + r_j - dist, 0)^2
    mu_o = mu
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu_o * viol ** 2
            if d < 1e-12:
                # degenerate: push apart arbitrarily
                grad[i, 0] += 2.0 * mu_o * viol * 1.0
                grad[j, 0] += 2.0 * mu_o * viol * (-1.0)
                grad[i, 2] += 2.0 * mu_o * viol * 1.0
                grad[j, 2] += 2.0 * mu_o * viol * 1.0
                continue
            # d viol / d cx_i = -dx/d, d viol/d cx_j = dx/d
            # d viol / d r_i = 1, d viol / d r_j = 1
            dv = 2.0 * mu_o * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    # clamp
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    # check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _grid_layout(kx, ky, W, H, jitter=0.0, seed=0):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    rng = np.random.RandomState(seed)
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            if jitter > 0:
                xx += rng.uniform(-jitter, jitter)
                yy += rng.uniform(-jitter, jitter)
                xx = np.clip(xx, 0, W)
                yy = np.clip(yy, 0, H)
            r = min(xx, yy, W - xx, H - yy) * 0.99
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        # pad with tiny circles at corners
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _zorder_layout(W, H, seed=0):
    """
    Generate a layout using Z-order curve to assign circles to grid cells.
    This ensures spatial locality in the initial configuration.
    """
    rng = np.random.RandomState(seed)
    # Use a grid slightly larger than N to allow some empty cells or merging
    side = int(np.ceil(np.sqrt(N))) + 1
    # Generate Z-order coordinates
    # For simplicity, we use a 2D grid and fill it in Z-order
    # We create a grid of size side x side
    grid_size = side * side
    # Create Z-order indices for the grid
    # A simple Z-order (Morton code) for 2D
    def morton_code(x, y):
        # Interleave bits
        result = 0
        for i in range(16): # Enough for small grids
            result |= ((x >> i) & 1) << (2 * i)
            result |= ((y >> i) & 1) << (2 * i + 1)
        return result
    
    # We want to place N circles. Let's pick N cells from the grid in Z-order.
    # First, generate all potential cell centers
    cell_centers = []
    for j in range(side):
        for i in range(side):
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            cell_centers.append((morton_code(i, j), cx, cy))
    
    # Sort by Z-order
    cell_centers.sort(key=lambda x: x[0])
    
    # Take the first N cells (or random N if we want diversity)
    # For a deterministic "Z-order" feel, we take the first N
    selected = cell_centers[:N]
    
    pts = []
    for _, cx, cy in selected:
        # Add some jitter to break symmetry
        cx += rng.uniform(-0.05, 0.05) * W / side
        cy += rng.uniform(-0.05, 0.05) * H / side
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
        
        # Initial radius estimate
        r = min(cx, cy, W - cx, H - cy) * 0.8
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
        
    return np.array(pts)

def _random_layout(W, H, seed=0):
    """Random feasible-ish layout."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        cx = rng.uniform(0.05 * W, 0.95 * W)
        cy = rng.uniform(0.05 * H, 0.95 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.3
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _optimize_from(x0, W, H, max_iter=200):
    """Run SLSQP from x0. Returns (best_x, best_sum)."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    def fun(x):
        v, _ = _objective_penalty(x, W, H)
        return v

    def grad(x):
        _, g = _objective_penalty(x, W, H)
        return g

    try:
        res = minimize(fun, best_x, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10,
                                'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Second pass from best_x with tighter tolerance
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 150, 'ftol': 1e-12,
                                     'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    candidates = []

    # Aspect ratios to try
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    # Grid sizes
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (6, 4), (4, 6)]
    
    # 1. Grid layouts with and without jitter
    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            # Plain grid
            g = _grid_layout(kx, ky, W, H, jitter=0.0, seed=42)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
            
            # Jittered grid
            g_jit = _grid_layout(kx, ky, W, H, jitter=0.05 * W / kx, seed=42)
            s_jit = _evaluate(g_jit, W, H)
            if s_jit is not None:
                candidates.append((s_jit, W, H, g_jit))
                
    # 2. Z-order layouts
    for W in Ws:
        H = 2.0 - W
        for seed in [42, 43, 44]:
            g_z = _zorder_layout(W, H, seed=seed)
            s_z = _evaluate(g_z, W, H)
            if s_z is not None:
                candidates.append((s_z, W, H, g_z))
                
    # 3. Random layouts
    for W in Ws:
        H = 2.0 - W
        for seed in [42, 43, 44, 45]:
            g_r = _random_layout(W, H, seed=seed)
            s_r = _evaluate(g_r, W, H)
            if s_r is not None:
                candidates.append((s_r, W, H, g_r))

    # Sort candidates by initial score
    candidates.sort(key=lambda t: -t[0])

    # Process top candidates
    top_k = min(15, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        remaining = DEADLINE - time.time()
        # Allocate more iterations if we have time
        max_iter = 200 if remaining > 60 else 100
        x_opt, s_opt = _optimize_from(g0, W, H, max_iter=max_iter)
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    # Final refinement on the best solution found
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        # Try a few random perturbations of the best solution
        n_extra = min(5, int(remaining // 10))
        for i in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0 = np.clip(x0, 0.0, None)
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _optimize_from(x0, best_W, best_H, max_iter=80)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    if best_arr is None:
        # Fallback: simple grid
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.99)
        best_arr = g
        best_W, best_H = W, H

    # Final cleanup and evaluation
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # If the best solution is invalid (shouldn't happen, but just in case),
        # try to repair or fall back to a known valid candidate
        for s0, W, H, g0 in candidates[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = np.maximum(g0[:, 2], 1e-6)
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.minimum(r2, lo2)
            r2 = np.maximum(r2, 1e-6)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        # Absolute fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
