"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize
the minimum triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0
DEADLINE = 115.0  # safety margin before 120s limit

# Precompute all triangle combinations
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])
n_combos = len(_combos)


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles formed by the points."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _soft_min(areas, temperature):
    """Soft minimum of areas using log-sum-exp trick (negative for min)."""
    # Use log-sum-exp for numerical stability
    shifted = areas / temperature
    max_val = np.max(shifted)
    if not np.isfinite(max_val):
        return 0.0
    log_sum = max_val + np.log(np.sum(np.exp(shifted - max_val)))
    return temperature * log_sum


def _objective_and_grad(flat, temperature):
    """Objective: negative soft-min of triangle areas. Returns (value, gradient)."""
    pts = flat.reshape(N, 2)
    
    # Project to triangle
    pts = _project_to_triangle(pts)
    
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # For soft-min, we use -soft_max of areas
    # soft_max(a) = t * log(sum(exp(a_i/t)))
    # We want to maximize min(areas), so minimize -soft_max(areas)
    shifted = areas / temperature
    max_val = np.max(shifted)
    if not np.isfinite(max_val):
        return 0.0, np.zeros_like(flat)
    log_sum = max_val + np.log(np.sum(np.exp(shifted - max_val)))
    soft_max_val = temperature * log_sum
    value = -soft_max_val
    
    # Gradient: d/dx_i of -soft_max(areas)
    # = -sum_j (exp(areas_j/t) / sum_k exp(areas_k/t)) * d(areas_j)/dx_i
    weights = np.exp(shifted - max_val)
    weights /= np.sum(weights)
    
    # d(areas_j)/d(pts) where areas_j = 0.5 * |cross_j|
    # For each triangle j with points i, j, k:
    # area = 0.5 * |(pj-pi) x (pk-pi)|
    # d(area)/dpi = -0.5 * sign(cross) * (pk-pj) rotated 90
    # d(area)/dpj = 0.5 * sign(cross) * (pk-pi) rotated 90
    # d(area)/dpk = -0.5 * sign(cross) * (pj-pi) rotated 90
    # where (a,b) rotated 90 = (-b, a)
    
    grad = np.zeros((N, 2))
    signs = np.sign(cross)
    # Handle zeros
    signs[signs == 0] = 1.0
    
    for j in range(n_combos):
        w = weights[j]
        if w < 1e-12:
            continue
        s = 0.5 * signs[j]
        a = _idx_i[j]
        b = _idx_j[j]
        c = _idx_k[j]
        pa, pb, pc = pts[a], pts[b], pts[c]
        
        # d(area)/dpa = -s * (pc - pb) rotated 90 = -s * (-(pc[1]-pb[1]), pc[0]-pb[0])
        # = s * (pc[1]-pb[1], -(pc[0]-pb[0]))
        dpa = s * np.array([pc[1] - pb[1], -(pc[0] - pb[0])])
        # d(area)/dpb = s * (pc - pa) rotated 90 = s * (-(pc[1]-pa[1]), pc[0]-pa[0])
        dpb = s * np.array([-(pc[1] - pa[1]), pc[0] - pa[0]])
        # d(area)/dpk = -s * (pb - pa) rotated 90 = -s * (-(pb[1]-pa[1]), pb[0]-pa[0])
        # = s * (pb[1]-pa[1], -(pb[0]-pa[0]))
        dpk = s * np.array([pb[1] - pa[1], -(pb[0] - pa[0])])
        
        grad[a] += w * dpa
        grad[b] += w * dpb
        grad[c] += w * dpk
    
    # Gradient of -soft_max is -sum(w_j * grad_area_j)
    grad_flat = -grad.flatten()
    
    return value, grad_flat


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle with vertices (0,0), (1,0), (0.5, H)."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Constraint 1: y >= 0
        if y < 0:
            y = 0
        # Constraint 2: y <= H
        if y > H:
            y = H
        # Constraint 3: x >= y / sqrt(3)  (left edge from (0,0) to (0.5, H))
        # The left edge: x = y / sqrt(3)
        min_x = y / np.sqrt(3)
        if x < min_x:
            x = min_x
        # Constraint 4: x <= 1 - y / sqrt(3)  (right edge from (1,0) to (0.5, H))
        max_x = 1.0 - y / np.sqrt(3)
        if x > max_x:
            x = max_x
        # Re-check y after x adjustment
        if y < 0:
            y = 0
        if y > H:
            y = H
        pts[i] = [x, y]
    return pts


def _initial_points(seed=0):
    """Generate initial points using a structured grid + jitter."""
    rng = np.random.RandomState(seed)
    # Use a grid-like initial placement
    # 11 points: maybe 3-4-4 or similar arrangement
    # Start with a rough grid and add random jitter
    base_points = []
    # Try a few structured initializations
    for trial in range(5):
        pts = []
        # Place points in a roughly uniform pattern
        rows = [1, 3, 3, 4]  # 11 points in 4 rows
        y_positions = [0.1, 0.3, 0.5, 0.7]
        for row, y_frac in enumerate(y_positions):
            y = y_frac * H
            n_in_row = rows[row]
            min_x = y / np.sqrt(3)
            max_x = 1.0 - y / np.sqrt(3)
            if n_in_row == 1:
                x = (min_x + max_x) / 2
                pts.append([x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)])
            else:
                for k in range(n_in_row):
                    frac = (k + 0.5) / n_in_row
                    x = min_x + frac * (max_x - min_x)
                    pts.append([x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)])
        pts = np.array(pts)
        pts = _project_to_triangle(pts)
        base_points.append(pts)
    
    # Pick the one with best min area
    best_pts = base_points[0]
    best_area = -1
    for pts in base_points:
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts
    
    return best_pts


def _run_l_bfgs_b(pts, temperature, maxiter=2000, time_limit=10.0):
    """Run L-BFGS-B optimization starting from pts."""
    start_time = time.time()
    flat = pts.flatten()
    
    def obj_and_grad(flat_x):
        val, grad = _objective_and_grad(flat_x, temperature)
        return val, grad
    
    result = minimize(
        obj_and_grad,
        flat,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': maxiter,
            'maxfun': 5000,
            'gtol': 1e-6,
            'ftol': 1e-8,
        }
    )
    
    elapsed = time.time() - start_time
    return result.x.reshape(N, 2), result.fun, elapsed


def construct():
    start_time = time.time()
    best_points = None
    best_score = -1
    
    # Multiple restarts with different initializations and temperatures
    seeds = list(range(20))
    temperatures = [0.05, 0.02, 0.01, 0.005]
    
    for seed in seeds:
        if time.time() - start_time > DEADLINE:
            break
        
        pts = _initial_points(seed)
        current_best = pts.copy()
        current_score = _min_triangle_area(pts)
        
        # Annealing: start with high temperature, decrease
        for temp in temperatures:
            if time.time() - start_time > DEADLINE:
                break
            
            # Try optimization from current best
            try:
                optimized, fun, elapsed = _run_l_bfgs_b(current_best, temp, maxiter=1000, time_limit=8.0)
                optimized = _project_to_triangle(optimized)
                score = _min_triangle_area(optimized)
                if score > current_score:
                    current_score = score
                    current_best = optimized
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_best.copy()
    
    # If we have time, do a final polish with very low temperature
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        try:
            polished, fun, elapsed = _run_l_bfgs_b(best_points, 0.001, maxiter=500, time_limit=5.0)
            polished = _project_to_triangle(polished)
            score = _min_triangle_area(polished)
            if score > best_score:
                best_points = polished
                best_score = score
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return initial points
        best_points = _initial_points(0)
    
    return np.array(best_points)
