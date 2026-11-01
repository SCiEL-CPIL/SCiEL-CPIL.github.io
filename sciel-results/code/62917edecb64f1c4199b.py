"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

N = 21

# EVOLVE-BLOCK-START

def _check_valid(x, y, r, tol=1e-7):
    """Check non-overlap and perimeter constraint."""
    n = len(r)
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    # Perimeter
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    bw = x_max - x_min
    bh = y_max - y_min
    if 2 * (bw + bh) > 4 + tol:
        return False
    return True


def _normalize(x, y, r):
    """Translate so bounding box min corner is at origin."""
    x_min = np.min(x - r)
    y_min = np.min(y - r)
    x -= x_min
    y -= y_min
    return x, y


def _force_relax(x, y, r, n_iter=300, lr=0.005, max_time=80.0, start_time=None):
    """Force-directed relaxation to resolve overlaps and enforce perimeter."""
    if start_time is None:
        start_time = time.time()
    n = len(r)
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_d = r[i] + r[j]
                if d < min_d:
                    f = (min_d - d) / d
                    fx[i] += f * dx
                    fy[i] += f * dy
                    fx[j] -= f * dx
                    fy[j] -= f * dy
        
        # Perimeter constraint: push inward if too large
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        perim = 2 * (bw + bh)
        if perim > 4:
            excess = perim - 4
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            for i in range(n):
                fx[i] -= 0.5 * excess * (x[i] - cx)
                fy[i] -= 0.5 * excess * (y[i] - cy)
        
        x += lr * fx
        y += lr * fy
        lr *= 0.995
    return x, y


def _expand_radii(x, y, r, max_time=60.0, start_time=None):
    """Greedily expand radii."""
    if start_time is None:
        start_time = time.time()
    n = len(r)
    step = 0.0005
    for outer in range(200):
        if time.time() - start_time > max_time:
            break
        improved = False
        for i in range(n):
            new_r = r.copy()
            new_r[i] += step
            if _check_valid(x, y, new_r):
                r = new_r
                improved = True
                break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return r


def _slsqp_refine(x, y, r, max_time=60.0, start_time=None):
    """SLSQP optimization on positions and radii."""
    if start_time is None:
        start_time = time.time()
    n = len(r)
    
    def objective(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        # Objective: negative sum of radii + penalties
        val = -np.sum(r)
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                pen = r[i] + r[j] - d
                if pen > 0:
                    val += 50 * pen * pen
        # Perimeter penalty
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        perim = 2 * ((x_max - x_min) + (y_max - y_min))
        if perim > 4:
            val += 50 * (perim - 4) * (perim - 4)
        return val
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 2.0))  # x
    for _ in range(n):
        bounds.append((0.0, 2.0))  # y
    for _ in range(n):
        bounds.append((0.01, 0.5))  # r
    
    v0 = np.concatenate([x, y, r])
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-8})
        if res.success or (time.time() - start_time < max_time):
            x_opt = res.x[:n]
            y_opt = res.x[n:2*n]
            r_opt = np.maximum(res.x[2*n:3*n], 0.005)
            return x_opt, y_opt, r_opt
    except Exception:
        pass
    return None


def _init_grid(cols, rows):
    """Grid initialization in a w x h rectangle (w+h=2)."""
    # Choose aspect ratio
    h = 1.0
    w = 1.0
    # Adjust for number of circles
    if cols * rows < N:
        # Need more cells
        cols = int(np.ceil(np.sqrt(N * 1.5)))
        rows = int(np.ceil(N / cols))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= N:
                break
            xs[count] = (i + 0.5) * w / cols
            ys[count] = (j + 0.5) * h / rows
            count += 1
    
    r = np.full(N, min(w / cols, h / rows) * 0.42)
    return xs[:N], ys[:N], r[:N]


def _init_random():
    """Random initialization."""
    xs = np.random.rand(N) * 1.0
    ys = np.random.rand(N) * 1.0
    r = np.random.rand(N) * 0.05 + 0.03
    return xs, ys, r


def _pack(max_time=140.0):
    start_time = time.time()
    best_result = None
    best_value = -np.inf
    
    def evaluate(x, y, r):
        if not _check_valid(x, y, r):
            return None, -np.inf
        val = np.sum(r)
        return np.column_stack([x, y, r]), val
    
    # Multiple initializations
    inits = [
        lambda: _init_grid(5, 5),
        lambda: _init_grid(6, 4),
        lambda: _init_grid(7, 3),
        lambda: _init_grid(4, 6),
        lambda: _init_random(),
        lambda: _init_random(),
    ]
    
    for init_func in inits:
        if time.time() - start_time > max_time - 15:
            break
        try:
            x, y, r = init_func()
            
            # Force relax
            x, y = _force_relax(x, y, r, n_iter=200, lr=0.004,
                                max_time=max_time - 30, start_time=start_time)
            
            # Normalize
            x, y = _normalize(x, y, r)
            
            # Expand radii
            r = _expand_radii(x, y, r, max_time=max_time - 25, start_time=start_time)
            
            # SLSQP refine
            if time.time() - start_time < max_time - 10:
                res = _slsqp_refine(x, y, r, max_time=max_time - 15, start_time=start_time)
                if res is not None:
                    x, y, r = res
                    x, y = _normalize(x, y, r)
            
            # Final relax + expand
            x, y = _force_relax(x, y, r, n_iter=100, lr=0.002,
                                max_time=max_time - 10, start_time=start_time)
            x, y = _normalize(x, y, r)
            r = _expand_radii(x, y, r, max_time=max_time - 5, start_time=start_time)
            
            result, val = evaluate(x, y, r)
            if result is not None and val > best_value:
                best_value = val
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: 5x5 grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                rr = 0.5 / side - 1e-4
                cells.append([cx, cy, rr])
        best_result = np.array(cells[:N])
    
    return best_result


def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    return _pack(max_time=140.0)

# EVOLVE-BLOCK-END
