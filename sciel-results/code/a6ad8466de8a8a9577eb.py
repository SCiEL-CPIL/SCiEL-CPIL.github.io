"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Structural change: SOCP-based formulation with direct constraint enforcement,
multi-stage refinement of container geometry, and jiggling polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(cx, cy, r, W, H):
    if _is_valid(cx, cy, r, W, H):
        return float(np.sum(r))
    return -1.0


def _expand_radii(x, y, r, w, h, max_iter=200):
    """Iteratively expand each radius to its maximum allowed value."""
    r = r.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.001)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return r


def _jiggle(x, y, r, w, h, max_iter=500, time_budget=5.0):
    """Random perturbation + local optimization to escape local minima."""
    t0 = time.time()
    best_r = r.copy()
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return r

    rng = np.random.default_rng(123)
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # Small random perturbation
        eps = 0.002 * (1.0 - it / max_iter)
        x_p = x + rng.uniform(-eps, eps, N)
        y_p = y + rng.uniform(-eps, eps, N)
        
        # Clamp to boundaries
        x_p = np.clip(x_p, r, w - r)
        y_p = np.clip(y_p, r, h - r)
        
        # Try to expand radii from perturbed positions
        r_p = _expand_radii(x_p, y_p, r, w, h)
        
        val = _sum_radii(x_p, y_p, r_p, w, h)
        if val > best_val:
            best_val = val
            best_r = r_p.copy()
            x = x_p.copy()
            y = y_p.copy()
            r = r_p.copy()
    
    return best_r


def _soptimize_positions(x, y, r, w, h, time_budget=10.0):
    """Optimize positions given fixed radii using SLSQP with explicit constraints."""
    t0 = time.time()
    n = N
    
    def objective(p):
        if time.time() - t0 > time_budget:
            return 0.0
        return 0.0  # We want to find feasible positions; use penalty for infeasibility
    
    # Actually, let's use a proper approach: minimize max overlap
    def objective_max_overlap(p):
        if time.time() - t0 > time_budget:
            return 1e8
        xi = p[:n]
        yi = p[n:2*n]
        max_ov = 0.0
        for i in range(n):
            max_ov = max(max_ov, r[i] - xi[i], xi[i] + r[i] - w, r[i] - yi[i], yi[i] + r[i] - h)
            for j in range(i + 1, n):
                dx = xi[i] - xi[j]
                dy = yi[i] - yi[j]
                d = np.sqrt(dx*dx + dy*dy)
                ov = r[i] + r[j] - d
                max_ov = max(max_ov, ov)
        return max_ov
    
    bounds = [(r[i], w - r[i]) for i in range(n)] + [(r[i], h - r[i]) for i in range(n)]
    
    try:
        res = minimize(objective_max_overlap, np.concatenate([x, y]), method='SLSQP',
                       bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 100})
        if res.fun < 1e-8:
            x_opt = res.x[:n]
            y_opt = res.x[n:2*n]
            return x_opt, y_opt
    except Exception:
        pass
    return x, y


def _optimize_aspect_ratio(x, y, r, w, h, time_budget=8.0):
    """Try to improve the aspect ratio by adjusting w and re-optimizing."""
    t0 = time.time()
    best = (x.copy(), y.copy(), r.copy(), w, h)
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return best
    
    # Try different aspect ratios
    for w_new in [w * 0.9, w * 1.1, w - 0.1, w + 0.1]:
        if time.time() - t0 > time_budget:
            break
        h_new = 2.0 - w_new
        if h_new < 0.3 or w_new < 0.3:
            continue
        
        # Scale positions
        scale_x = w_new / w
        scale_y = h_new / h
        x_new = x * scale_x
        y_new = y * scale_y
        
        # Try to expand radii
        r_new = _expand_radii(x_new, y_new, r, w_new, h_new)
        val = _sum_radii(x_new, y_new, r_new, w_new, h_new)
        if val > best_val:
            best_val = val
            best = (x_new.copy(), y_new.copy(), r_new.copy(), w_new, h_new)
    
    return best


def _full_refine(x, y, r, w, h, time_budget=20.0):
    """Full refinement: optimize positions, expand radii, try aspect ratios."""
    t0 = time.time()
    best_val = _sum_radii(x, y, r, w, h)
    if best_val < 0:
        return x, y, r, w, h
    
    for _ in range(3):
        if time.time() - t0 > time_budget:
            break
        
        # Optimize positions
        x, y = _soptimize_positions(x, y, r, w, h, time_budget=min(5.0, time_budget - (time.time() - t0)))
        
        # Expand radii
        r = _expand_radii(x, y, r, w, h)
        
        # Try aspect ratio
        x, y, r, w, h = _optimize_aspect_ratio(x, y, r, w, h, time_budget=min(5.0, time_budget - (time.time() - t0)))
        
        val = _sum_radii(x, y, r, w, h)
        if val > best_val:
            best_val = val
    
    # Final jiggle
    if time.time() - t0 < time_budget - 3.0:
        r = _jiggle(x, y, r, w, h, max_iter=300, time_budget=max(1.0, time_budget - (time.time() - t0)))
        # Re-expand after jiggle
        r = _expand_radii(x, y, r, w, h)
    
    return x, y, r, w, h


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_value = 0.0
    best_x, best_y, best_r, best_w, best_h = None, None, None, None, None
    
    def get_initial_guesses():
        guesses = []
        
        # Grid-based initializations with different aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.25, 0.8, 1.4, 0.7, 1.1, 0.9, 1.3, 0.78]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue
            
            # Try different grid configurations
            for n_rows, n_cols in [(4, 6), (5, 5), (3, 7), (6, 4), (7, 3), (4, 5), (5, 4)]:
                if n_rows * n_cols < N:
                    continue
                if n_rows * n_cols > N + 4:
                    continue
                
                x = np.zeros(N)
                y = np.zeros(N)
                r = np.zeros(N)
                
                cell_w = w0 / n_cols
                cell_h = h0 / n_rows
                
                count = 0
                for row in range(n_rows):
                    for col in range(n_cols):
                        if count >= N:
                            break
                        cx = (col + 0.5) * cell_w
                        cy = (row + 0.5) * cell_h
                        r0 = min(cell_w, cell_h) * 0.44
                        x[count] = cx
                        y[count] = cy
                        r[count] = r0
                        count += 1
                
                guesses.append((x, y, r, w0, h0))
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                x[i] = rng.uniform(r0, w0 - r0)
                y[i] = rng.uniform(r0, h0 - r0)
                r[i] = r0
            guesses.append((x, y, r, w0, h0))
        
        return guesses
    
    guesses = get_initial_guesses()
    
    # Sort guesses by initial potential (sum of radii)
    def potential(g):
        x, y, r, w, h = g
        # Quick check
        if not _is_valid(x, y, r, w, h):
            # Estimate potential
            return np.sum(r) * 0.5
        return np.sum(r)
    
    guesses.sort(key=potential, reverse=True)
    
    for i, (x, y, r, w, h) in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        # Quick validity check
        if not _is_valid(x, y, r, w, h):
            # Try to fix by shrinking
            r_fix = r * 0.9
            if not _is_valid(x, y, r_fix, w, h):
                continue
        
        # Refine
        x_r, y_r, r_r, w_r, h_r = _full_refine(x, y, r, w, h, time_budget=12.0)
        
        val = _sum_radii(x_r, y_r, r_r, w_r, h_r)
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x_r.copy(), y_r.copy(), r_r.copy(), w_r, h_r
    
    # Final polish on best result
    if best_x is not None and time.time() < deadline - 10:
        x_f, y_f, r_f, w_f, h_f = _full_refine(best_x, best_y, best_r, best_w, best_h, time_budget=15.0)
        val = _sum_radii(x_f, y_f, r_f, w_f, h_f)
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x_f.copy(), y_f.copy(), r_f.copy(), w_f, h_f
    
    # Fallback
    if best_x is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        best_x = np.zeros(N)
        best_y = np.zeros(N)
        best_r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, side)
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 0.01
        best_w, best_h = w, h
    
    # Final safety shrink
    for _ in range(30):
        r_trial = best_r * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h):
            best_r = r_trial
        else:
            break
    
    return np.column_stack((best_x, best_y, best_r))
