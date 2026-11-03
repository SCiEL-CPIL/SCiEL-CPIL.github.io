"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21
DEADLINE = 150.0  # seconds; evaluator kills at 150s, keep safety margin
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _is_valid(circles, W, H, tol=1e-7):
    """Check all circles are inside [0,W]x[0,H] and non-overlapping."""
    if circles is None or len(circles) != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= -tol):
        return False
    # Containment
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    # Non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx*dx + dy*dy < (r[i] + r[j])**2 - tol:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _hex_lattice_positions(n, aspect):
    """Generate n circle centers in a hexagonal close-packed pattern,
    normalized so the bounding box has the given aspect ratio (W/H).
    Returns (centers, W, H) where W + H = 2 (perimeter 4)."""
    pts = []
    row = 0
    while len(pts) < n + 10:
        count = 12
        for i in range(count):
            x = i + (0.5 if row % 2 else 0.0)
            y = row * np.sqrt(3) / 2.0
            pts.append((x, y))
        row += 1
    pts = np.array(pts[:n])
    
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    if w < 1e-9: w = 1.0
    if h < 1e-9: h = 1.0
    pts = (pts - np.array([xmin, ymin])) / np.array([w, h])
    
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    pts = pts * np.array([W, H])
    
    return pts, W, H


def _relax(circles, W, H, max_passes=200, tol=1e-8):
    """Resolve overlaps and boundary violations via iterative relaxation.
    Returns a new valid (or best-effort) circles array."""
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_passes):
        if _time_left() < 3.0:
            break
        moved = False
        
        # Boundary clamp
        for i in range(n):
            r = circles[i, 2]
            if circles[i, 0] < r:
                circles[i, 0] = r
                moved = True
            if circles[i, 0] > W - r:
                circles[i, 0] = W - r
                moved = True
            if circles[i, 1] < r:
                circles[i, 1] = r
                moved = True
            if circles[i, 1] > H - r:
                circles[i, 1] = H - r
                moved = True
        
        # Pairwise collision resolution
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i, 2] + circles[j, 2]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-10:
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    circles[i, 0] -= nx * overlap * 0.5
                    circles[i, 1] -= ny * overlap * 0.5
                    circles[j, 0] += nx * overlap * 0.5
                    circles[j, 1] += ny * overlap * 0.5
                    moved = True
        
        if not moved:
            break
    
    return circles


def _grow_and_resolve(circles, W, H, growth_rate=0.002, max_steps=3000):
    """Iteratively grow radii and resolve collisions. Returns best valid config."""
    circles = circles.copy()
    best_circles = circles.copy()
    best_sum = _sum_radii(circles)
    
    for step in range(max_steps):
        if _time_left() < 3.0:
            break
        
        # Grow radii
        circles[:, 2] *= (1.0 + growth_rate)
        
        # Resolve
        circles = _relax(circles, W, H, max_passes=50)
        
        # Check validity
        if _is_valid(circles, W, H):
            s = _sum_radii(circles)
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()
        else:
            # Rollback growth
            circles[:, 2] /= (1.0 + growth_rate)
            # Try a smaller growth
            circles[:, 2] *= 1.0005
            circles = _relax(circles, W, H, max_passes=20)
            if not _is_valid(circles, W, H):
                circles[:, 2] /= 1.0005
                break
    
    return best_circles


def _scipy_polish(circles, W, H, time_budget=10.0):
    """Use scipy.optimize.minimize to polish positions and radii.
    Maximizes sum(r) subject to non-overlap and containment constraints via penalty."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return circles
    
    start_time = time.time()
    n = len(circles)
    x0 = circles.flatten()
    
    # Bounds
    max_r = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((0, max_r))
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return np.sum(x[2::3])  # stop early, return something
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        
        val = -np.sum(rs)
        penalty = 0.0
        pen_factor = 10000.0
        
        # Wall constraints
        for i in range(n):
            if xs[i] - rs[i] < 0:
                penalty += pen_factor * (xs[i] - rs[i])**2
            if xs[i] + rs[i] > W:
                penalty += pen_factor * (xs[i] + rs[i] - W)**2
            if ys[i] - rs[i] < 0:
                penalty += pen_factor * (ys[i] - rs[i])**2
            if ys[i] + rs[i] > H:
                penalty += pen_factor * (ys[i] + rs[i] - H)**2
        
        # Pairwise
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if dist < min_d:
                    penalty += pen_factor * (min_d - dist)**2
        
        return val + penalty
    
    def callback(xk):
        return time.time() - start_time > time_budget
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       callback=callback, options={'maxiter': 500, 'maxfun': 2000})
        if res.success or res.fun < objective(x0):
            x_opt = res.x.reshape(n, 3)
            # Clamp to valid range
            for i in range(n):
                x_opt[i, 0] = np.clip(x_opt[i, 0], 0, W)
                x_opt[i, 1] = np.clip(x_opt[i, 1], 0, H)
                x_opt[i, 2] = np.clip(x_opt[i, 2], 0, max_r)
            # Relax to ensure validity
            x_opt = _relax(x_opt, W, H, max_passes=100)
            if _is_valid(x_opt, W, H) and _sum_radii(x_opt) > _sum_radii(circles):
                return x_opt
    except Exception:
        pass
    
    return circles


def _try_aspect(aspect, time_budget=None):
    """Try packing with given aspect ratio. Returns best circles and sum."""
    if time_budget is None:
        time_budget = max(5.0, _time_left() * 0.3)
    
    centers, W, H = _hex_lattice_positions(N, aspect)
    r0 = 0.03
    circles = np.column_stack([centers[:, 0], centers[:, 1], np.full(N, r0)])
    
    # Phase 1: Grow and resolve
    result = _grow_and_resolve(circles, W, H, growth_rate=0.002, max_steps=2000)
    
    if not _is_valid(result, W, H):
        return None, 0.0
    
    # Phase 2: Scipy polish if time allows
    if _time_left() > 15.0:
        polished = _scipy_polish(result, W, H, time_budget=min(10.0, _time_left() * 0.2))
        if _is_valid(polished, W, H) and _sum_radii(polished) > _sum_radii(result):
            result = polished
    
    return result, _sum_radii(result)


def _random_initialization(W, H, n, seed=None):
    """Generate random initial circle positions."""
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    r0 = 0.02
    circles = np.zeros((n, 3))
    circles[:, 0] = np.random.uniform(r0, W - r0, n)
    circles[:, 1] = np.random.uniform(r0, H - r0, n)
    circles[:, 2] = r0
    return circles


def construct():
    """Return array of shape (21, 3): [x, y, r] for 21 disjoint circles
    in a rectangle of perimeter at most 4, maximizing sum of radii."""
    best_circles = None
    best_sum = 0.0
    
    # Try several aspect ratios
    aspects = [1.0, 1.2, 1.4, 1.6, 1.73, 1.8, 2.0, 0.8, 0.7, 0.6]
    
    for aspect in aspects:
        if _time_left() < 8.0:
            break
        circles, s = _try_aspect(aspect)
        if circles is not None and s > best_sum:
            best_sum = s
            best_circles = circles
    
    # If time remains, try a few random initializations with best aspect
    if best_circles is not None and _time_left() > 10.0:
        # Re-derive W, H from best_circles bounding box
        xs = best_circles[:, 0]
        ys = best_circles[:, 1]
        rs = best_circles[:, 2]
        W = float(np.max(xs + rs))
        H = float(np.max(ys + rs))
        
        for trial in range(3):
            if _time_left() < 8.0:
                break
            rand_circles = _random_initialization(W, H, N, seed=trial * 100 + 42)
            result = _grow_and_resolve(rand_circles, W, H, growth_rate=0.003, max_steps=1000)
            if _is_valid(result, W, H):
                s = _sum_radii(result)
                if s > best_sum:
                    best_sum = s
                    best_circles = result
    
    if best_circles is None:
        # Fallback: seed-like square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
