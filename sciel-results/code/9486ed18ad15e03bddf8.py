"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _bounding_rect(circles):
    """Return (w, h) of bounding rectangle of circles."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    return x_max - x_min, y_max - y_min


def _perimeter(circles):
    w, h = _bounding_rect(circles)
    return 2.0 * (w + h)


def _sum_radii(circles):
    return np.sum(circles[:, 2])


def _is_valid(circles, tol=1e-6):
    """Check that circles are disjoint and perimeter <= 4 + tol."""
    if _perimeter(circles) > 4.0 + tol:
        return False
    # Check pairwise disjointness
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True


def _objective(params, n):
    """Penalty objective for optimizer. params = [x0,y0,r0, x1,y1,r1, ...]"""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Enforce positive radii
    r = np.maximum(r, 1e-6)
    
    # Perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    w = x_max - x_min
    h = y_max - y_min
    
    # We want to maximize sum(r), so minimize -sum(r) with penalties
    obj = -np.sum(r)
    
    # Perimeter penalty
    perim_viol = max(0.0, 2.0 * (w + h) - 4.0)
    obj += 1e4 * perim_viol ** 2
    
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                obj += 1e4 * overlap ** 2
    
    return obj


def _grid_initialization(n, seed=42):
    """Generate a reasonable grid-based initial packing."""
    rng = np.random.RandomState(seed)
    
    # Try different grid arrangements
    best_circles = None
    best_obj = np.inf
    
    for cols in range(3, 8):
        rows = int(np.ceil(n / cols))
        # Create grid positions
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= n:
                    break
                cx = (i + 0.5) / cols
                cy = (j + 0.5) / rows
                # Estimate radius based on spacing
                r = min(0.5 / cols, 0.5 / rows) * 0.95
                circles.append([cx, cy, r])
        circles = np.array(circles[:n])
        
        # Scale to fit perimeter
        obj = _objective(circles.ravel(), n)
        if obj < best_obj:
            best_obj = obj
            best_circles = circles.copy()
    
    return best_circles


def _run_optimization(initial_circles, start_time, max_time_per_run=10.0):
    """Run scipy optimization from given initial point."""
    n = len(initial_circles)
    x0 = initial_circles.ravel()
    
    # Bounds: positions in [0,1], radii in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    best_circles = initial_circles.copy()
    best_obj = _objective(x0, n)
    
    elapsed = time.time() - start_time
    remaining = max_time_per_run - elapsed
    if remaining <= 0:
        return best_circles, best_obj
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective, x0,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_circles = result.x.reshape(n, 3)
    except Exception:
        pass
    
    # Try Nelder-Mead on a perturbed version
    elapsed = time.time() - start_time
    remaining = max_time_per_run - elapsed
    if remaining > 2.0:
        try:
            # Perturb slightly
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.randn(n) * 0.01
            perturbed[:, 1] += np.random.randn(n) * 0.01
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0.01, 0.99)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.01, 0.99)
            
            result2 = minimize(
                _objective, perturbed.ravel(),
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_circles = result2.x.reshape(n, 3)
        except Exception:
            pass
    
    return best_circles, best_obj


def _normalize_circles(circles):
    """Translate circles so bounding box is in [0,w]x[0,h] and scale to perimeter=4."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    x_min = np.min(x - r)
    y_min = np.min(y - r)
    x_max = np.max(x + r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    if w < 1e-10 or h < 1e-10:
        return circles
    
    # Scale so perimeter = 4: 2*(w*h_scale) = 4 => w_scale = 2/(w+h)
    scale = 2.0 / (w + h)
    
    # Translate and scale
    circles_out = circles.copy()
    circles_out[:, 0] = (x - x_min) * scale
    circles_out[:, 1] = (y - y_min) * scale
    circles_out[:, 2] = r * scale
    
    return circles_out


def construct():
    start_time = time.time()
    
    # Get initial packing
    initial = _grid_initialization(N, seed=42)
    
    best_circles = initial.copy()
    best_sum_r = _sum_radii(best_circles)
    
    # Run optimization from initial
    if time.time() - start_time < DEADLINE - 5.0:
        circles, obj = _run_optimization(initial, start_time, max_time_per_run=15.0)
        if _sum_radii(circles) > best_sum_r:
            best_circles = circles.copy()
            best_sum_r = _sum_radii(best_circles)
    
    # Try multiple random restarts if time permits
    rng = np.random.RandomState(123)
    attempt = 0
    while time.time() - start_time < DEADLINE - 10.0:
        attempt += 1
        
        # Generate random initial configuration
        cols = rng.randint(3, 7)
        rows = int(np.ceil(N / cols))
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) / cols + rng.randn() * 0.05
                cy = (j + 0.5) / rows + rng.randn() * 0.05
                r = min(0.5 / cols, 0.5 / rows) * (0.8 + rng.rand() * 0.3)
                circles.append([np.clip(cx, 0.05, 0.95), np.clip(cy, 0.05, 0.95), r])
        circles = np.array(circles[:N])
        
        # Run quick optimization
        if time.time() - start_time < DEADLINE - 8.0:
            c_opt, obj = _run_optimization(circles, start_time, max_time_per_run=8.0)
            if _sum_radii(c_opt) > best_sum_r:
                best_circles = c_opt.copy()
                best_sum_r = _sum_radii(best_circles)
        
        # Also try with different aspect ratio bias
        if time.time() - start_time < DEADLINE - 10.0 and attempt % 3 == 0:
            # Try elongated arrangement
            for aspect in [0.5, 1.0, 2.0]:
                if time.time() - start_time >= DEADLINE - 10.0:
                    break
                circles2 = []
                for i in range(N):
                    cx = rng.uniform(0.1, 0.9)
                    cy = rng.uniform(0.1, 0.9)
                    r = 0.05 + rng.rand() * 0.1
                    circles2.append([cx, cy, r])
                circles2 = np.array(circles2)
                
                if time.time() - start_time < DEADLINE - 8.0:
                    c_opt2, obj2 = _run_optimization(circles2, start_time, max_time_per_run=6.0)
                    if _sum_radii(c_opt2) > best_sum_r:
                        best_circles = c_opt2.copy()
                        best_sum_r = _sum_radii(c_opt2)
    
    # Final refinement with more iterations if time allows
    if time.time() - start_time < DEADLINE - 5.0:
        try:
            n = len(best_circles)
            x0 = best_circles.ravel()
            bounds = []
            for i in range(n):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.001, 0.5))
            
            result = minimize(
                _objective, x0,
                args=(n,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < _objective(x0, n):
                refined = result.x.reshape(n, 3)
                if _sum_radii(refined) > best_sum_r:
                    best_circles = refined
                    best_sum_r = _sum_radii(refined)
        except Exception:
            pass
    
    # Ensure validity: if perimeter > 4, scale down
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    # Ensure no overlaps by slight shrink if needed
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_circles[i, 0] - best_circles[j, 0]
            dy = best_circles[i, 1] - best_circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = best_circles[i, 2] + best_circles[j, 2]
            if dist < min_dist:
                # Shrink both slightly
                factor = dist / min_dist * 0.999
                best_circles[i, 2] *= factor
                best_circles[j, 2] *= factor
    
    # Final perimeter check
    w, h = _bounding_rect(best_circles)
    perim = 2.0 * (w + h)
    if perim > 4.0:
        scale = 4.0 / perim * 0.9999
        best_circles[:, 0] *= scale
        best_circles[:, 1] *= scale
        best_circles[:, 2] *= scale
    
    return best_circles
