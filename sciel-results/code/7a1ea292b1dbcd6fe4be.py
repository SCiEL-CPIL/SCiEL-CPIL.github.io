"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _validate(circles):
    """Check disjoint interiors and perimeter <= 4. Returns True if valid."""
    if len(circles) != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return False
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return False
    if 2 * (w + h) > 4.0 + 1e-9:
        return False
    # Disjoint interiors: distance between centers >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True


def _penalty(circles):
    """Penalty for constraint violations (0 if valid)."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return 1e6
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    pen = 0.0
    if w <= 0 or h <= 0:
        return 1e6
    perim = 2 * (w + h)
    if perim > 4.0:
        pen += (perim - 4.0) ** 2 * 100
    # Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            gap = rs[i] + rs[j] - dist
            if gap > 0:
                pen += gap ** 2 * 100
    return pen


def _score(circles):
    """Sum of radii, or -inf if invalid."""
    if not _validate(circles):
        return -np.inf
    return np.sum(circles[:, 2])


def _normalize(circles):
    """Translate and scale so bounding box has perimeter exactly 4 (if possible)."""
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return circles
    # Scale to make perimeter = 4
    scale = 2.0 / (w + h)
    xs = (xs - min_x) * scale
    ys = (ys - min_y) * scale
    rs = rs * scale
    return np.column_stack([xs, ys, rs])


def _grid_init(w, h, n_circles=21):
    """Create a grid of circles in a rectangle of width w, height h."""
    cols = int(np.ceil(np.sqrt(n_circles * w / h))) if w > h else int(np.ceil(np.sqrt(n_circles * h / w)))
    cols = max(1, min(cols, n_circles))
    rows = int(np.ceil(n_circles / cols))
    cells = []
    for i in range(cols):
        for j in range(rows):
            if len(cells) >= n_circles:
                break
            cx = (i + 0.5) * w / cols
            cy = (j + 0.5) * h / rows
            r = min(w / cols, h / rows) * 0.48
            cells.append([cx, cy, r])
    return np.array(cells[:n_circles])


def _optimize(circles, max_iter=200, tol=1e-8):
    """Use scipy minimize to maximize sum of radii subject to constraints."""
    n = len(circles)
    
    def objective(params):
        c = params.reshape(3, -1).T
        # We want to maximize sum of radii, so minimize negative
        # But also penalize constraint violations
        pen = _penalty(c)
        return -np.sum(c[:, 2]) + pen
    
    x0 = circles.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
    best = result.x.reshape(3, -1).T
    return best


def _local_search(circles, steps=5000, step_size=0.01):
    """Simple local search with random perturbations."""
    best = circles.copy()
    best_score = _score(best)
    if not np.isfinite(best_score):
        best_score = -np.inf
    
    rng = np.random.default_rng(42)
    for step in range(steps):
        if _elapsed() > _DEADLINE:
            break
        trial = best + rng.normal(0, step_size, size=best.shape)
        # Ensure radii positive
        trial[:, 2] = np.maximum(trial[:, 2], 1e-4)
        s = _score(trial)
        if s > best_score:
            best = trial
            best_score = s
    return best


def construct():
    best_circles = None
    best_score = -np.inf
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.25, 2.5, 0.5, 1.75]
    
    for ar in aspect_ratios:
        if _elapsed() > _DEADLINE:
            break
        # Perimeter = 4, so 2(w+h) = 4 => w+h = 2
        # w/h = ar => w = ar*h => ar*h + h = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Grid initialization
        circles = _grid_init(w, h, N)
        
        # Normalize to ensure perimeter <= 4
        circles = _normalize(circles)
        
        # Validate
        if _validate(circles):
            s = _score(circles)
            if s > best_score:
                best_score = s
                best_circles = circles.copy()
        
        # Optimize with scipy
        if _elapsed() > _DEADLINE:
            break
        try:
            opt = _optimize(circles, max_iter=100)
            opt = _normalize(opt)
            if _validate(opt):
                s = _score(opt)
                if s > best_score:
                    best_score = s
                    best_circles = opt.copy()
        except Exception:
            pass
        
        # Local search
        if _elapsed() > _DEADLINE:
            break
        try:
            ls = _local_search(opt if _validate(opt) else circles, steps=2000, step_size=0.005)
            ls = _normalize(ls)
            if _validate(ls):
                s = _score(ls)
                if s > best_score:
                    best_score = s
                    best_circles = ls.copy()
        except Exception:
            pass
    
    # Fallback to simple grid if nothing valid
    if best_circles is None or not np.isfinite(best_score):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
