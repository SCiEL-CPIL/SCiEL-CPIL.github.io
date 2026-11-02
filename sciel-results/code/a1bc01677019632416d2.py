"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Lagrangian Relaxation with Subgradient Optimization.
- Treat the 165 area constraints as soft penalties with dual variables (Lagrange multipliers).
- Dual ascent: update multipliers based on constraint violation.
- Primal step: minimize the Lagrangian (quadratic-ish objective) using L-BFGS-B.
- This avoids non-smoothness of hard min and allows global exploration via penalty weights.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian coordinates."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian coordinates to barycentric coordinates."""
    b2 = 2 * pts[:, 1] / np.sqrt(3)
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _all_areas(pts):
    """Compute all 165 triangle areas. Returns (areas_array, triplet_indices)."""
    n = len(pts)
    areas = []
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                triplets.append((i, j, k))
    return np.array(areas), triplets

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _lagrangian_objective(bary, multipliers, area_target):
    """
    Compute the Lagrangian: sum of multipliers * max(0, area_target - area_m)
    plus a small regularization term.
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = _bary_to_cart(bary)
    areas, triplets = _all_areas(pts)
    
    # Constraint violation: max(0, area_target - area_m)
    violations = np.maximum(0, area_target - areas)
    
    # Lagrangian value
    L_val = np.sum(multipliers * violations)
    
    # Gradient computation
    # d(L)/d(bary) = sum_m multipliers[m] * d(violation_m)/d(bary)
    # d(violation_m)/d(bary) is nonzero only if area_m < area_target
    # For active constraints (area_m < area_target):
    #   d(violation_m)/d(area_m) = -1
    #   d(area_m)/d(bary) computed via chain rule
    
    grad_bary = np.zeros_like(bary)
    
    for m, (i, j, k) in enumerate(triplets):
        if violations[m] == 0:
            continue
        w = multipliers[m]
        if w == 0:
            continue
        
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        
        dx1 = xj - xi
        dy1 = yj - yi
        dx2 = xk - xi
        dy2 = yk - yi
        cross = dx1 * dy2 - dy1 * dx2
        
        # d(area_m)/d(bary)
        # area = 0.5 * |cross|
        # If cross > 0: d(area)/d(x_i) = -0.5*dy2, d(area)/d(y_i) = 0.5*dx2, etc.
        # If cross < 0: signs flip
        
        if cross >= 0:
            darea_dx_i = -0.5 * dy2
            darea_dy_i = 0.5 * dx2
            darea_dx_j = 0.5 * dy2
            darea_dy_j = -0.5 * dx2
            darea_dx_k = -0.5 * dy1
            darea_dy_k = 0.5 * dx1
        else:
            darea_dx_i = 0.5 * dy2
            darea_dy_i = -0.5 * dx2
            darea_dx_j = -0.5 * dy2
            darea_dy_j = 0.5 * dx2
            darea_dx_k = 0.5 * dy1
            darea_dy_k = -0.5 * dx1
        
        # d(violation)/d(bary) = -d(area)/d(bary) for active constraints
        # Convert Cartesian gradient to barycentric:
        # grad_b0 = 0
        # grad_b1 = grad_x
        # grad_b2 = 0.5*grad_x + sqrt(3)/2*grad_y
        
        # For point i
        grad_bary[i, 1] += w * (-darea_dx_i)
        grad_bary[i, 2] += w * (-0.5 * darea_dx_i - (np.sqrt(3) / 2) * darea_dy_i)
        # For point j
        grad_bary[j, 1] += w * (-darea_dx_j)
        grad_bary[j, 2] += w * (-0.5 * darea_dx_j - (np.sqrt(3) / 2) * darea_dy_j)
        # For point k
        grad_bary[k, 1] += w * (-darea_dx_k)
        grad_bary[k, 2] += w * (-0.5 * darea_dx_k - (np.sqrt(3) / 2) * darea_dy_k)
    
    return L_val, grad_bary

def _solve_primal(bary_init, multipliers, area_target, start_time, deadline):
    """Solve the primal problem using L-BFGS-B."""
    # Flatten barycentric coordinates
    x0 = bary_init.flatten()
    
    def objective(x):
        bary = x.reshape(N, 3)
        val, grad = _lagrangian_objective(bary, multipliers, area_target)
        return val, grad.flatten()
    
    # Bounds: barycentric coordinates must be in [0, 1]
    bounds = [(0, 1)] * (N * 3)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 50, 'maxfun': 500, 'disp': False})
        return result.x.reshape(N, 3)
    except Exception:
        return bary_init

def _dual_ascent(multipliers, areas, area_target, step_size):
    """Update Lagrange multipliers via subgradient method."""
    violations = np.maximum(0, area_target - areas)
    new_multipliers = multipliers + step_size * violations
    # Keep multipliers non-negative
    new_multipliers = np.maximum(0, new_multipliers)
    return new_multipliers

def _lagrangian_optimize(rng, start_time, deadline):
    """
    Main Lagrangian Relaxation optimization loop.
    """
    best_bary = None
    best_score = 0
    
    # Initialize with a reasonable starting configuration
    bary = np.zeros((N, 3))
    bary[0] = [1, 0, 0]  # vertex 0
    bary[1] = [0, 1, 0]  # vertex 1
    bary[2] = [0, 0, 1]  # vertex 2
    bary[3] = [0.5, 0.5, 0]  # midpoint of edge 0-1
    bary[4] = [0.5, 0, 0.5]  # midpoint of edge 0-2
    bary[5] = [0, 0.5, 0.5]  # midpoint of edge 1-2
    bary[6] = [1/3, 1/3, 1/3]  # center
    for i in range(7, N):
        bary[i] = np.array([rng.random(), rng.random(), rng.random()])
        bary[i] /= bary[i].sum()
    
    # Initial area target
    pts = _bary_to_cart(bary)
    areas, _ = _all_areas(pts)
    area_target = np.min(areas) * 1.1  # Start slightly above current min
    
    # Initialize multipliers
    multipliers = np.ones(len(areas)) * 0.1
    
    # Dual step size
    dual_step = 0.01
    dual_decay = 0.99
    
    # Main loop
    max_dual_iters = 500
    for dual_iter in range(max_dual_iters):
        if time.time() - start_time > deadline - 10:
            break
        
        # Primal step: minimize Lagrangian
        bary = _solve_primal(bary, multipliers, area_target, start_time, deadline)
        
        # Ensure barycentric coordinates are valid
        for i in range(N):
            s = bary[i].sum()
            if s > 0:
                bary[i] /= s
            else:
                bary[i] = np.array([1/3, 1/3, 1/3])
            bary[i] = np.clip(bary[i], 0, 1)
            s = bary[i].sum()
            if s > 0:
                bary[i] /= s
        
        # Evaluate current configuration
        pts = _bary_to_cart(bary)
        areas, _ = _all_areas(pts)
        current_min = np.min(areas)
        
        if current_min > best_score:
            best_score = current_min
            best_bary = bary.copy()
        
        # Dual ascent: update multipliers
        multipliers = _dual_ascent(multipliers, areas, area_target, dual_step)
        
        # Update area target: increase if we're making progress
        avg_area = np.mean(areas)
        if current_min > area_target * 0.95:
            area_target = min(avg_area * 0.8, current_min * 1.05)
        
        # Decay dual step size
        dual_step *= dual_decay
        
        # Occasionally perturb to escape local minima
        if dual_iter % 50 == 49:
            for i in range(N):
                bary[i] += rng.gauss(0, 0.01)
                bary[i] = np.clip(bary[i], 0, 1)
                bary[i] /= bary[i].sum()
    
    return best_bary, best_score

def _multi_start_lagrangian(rng, start_time, deadline):
    """Multi-start with Lagrangian Relaxation."""
    best_bary = None
    best_score = 0
    
    n_starts = 3
    
    for start_idx in range(n_starts):
        if time.time() - start_time > deadline - 20:
            break
        
        # Different initial configurations
        if start_idx == 0:
            bary_init = np.zeros((N, 3))
            bary_init[0] = [1, 0, 0]
            bary_init[1] = [0, 1, 0]
            bary_init[2] = [0, 0, 1]
            bary_init[3] = [0.5, 0.5, 0]
            bary_init[4] = [0.5, 0, 0.5]
            bary_init[5] = [0, 0.5, 0.5]
            bary_init[6] = [1/3, 1/3, 1/3]
            for i in range(7, N):
                bary_init[i] = np.array([rng.random(), rng.random(), rng.random()])
                bary_init[i] /= bary_init[i].sum()
        elif start_idx == 1:
            bary_init = np.array([np.array([rng.random(), rng.random(), rng.random()]) for _ in range(N)])
            bary_init = bary_init / bary_init.sum(axis=1, keepdims=True)
        else:
            # Golden ratio spiral-like initialization
            bary_init = np.zeros((N, 3))
            for i in range(N):
                angle = i * 2.39996  # Golden angle
                r = 0.3 * np.sqrt(i + 1) / np.sqrt(N)
                bary_init[i, 0] = 0.5 * (1 - r) + 0.25 * r * np.cos(angle)
                bary_init[i, 1] = 0.5 * (1 - r) + 0.25 * r * np.cos(angle + 2*np.pi/3)
                bary_init[i, 2] = 0.5 * (1 - r) + 0.25 * r * np.cos(angle + 4*np.pi/3)
                bary_init[i] /= bary_init[i].sum()
        
        # Run Lagrangian optimization with this start
        bary, score = _lagrangian_optimize(rng, start_time, deadline)
        
        if bary is not None and score > best_score:
            best_score = score
            best_bary = bary
    
    return best_bary, best_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Lagrangian Relaxation with Subgradient Optimization.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Multi-start Lagrangian optimization
    best_bary, best_score = _multi_start_lagrangian(rng, start_time, DEADLINE)
    
    if best_bary is None:
        # Fallback: random configuration
        best_bary = np.array([np.array([rng.random(), rng.random(), rng.random()]) for _ in range(N)])
        best_bary = best_bary / best_bary.sum(axis=1, keepdims=True)
        best_score = _min_triangle_area(_bary_to_cart(best_bary))
    
    # Convert to Cartesian coordinates
    best_pts = _bary_to_cart(best_bary)
    
    # Project to ensure we're inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    # Check validity
    if not _check_inside(best_pts):
        # Fallback to a known valid configuration
        best_pts = np.array([
            [0, 0],
            [1, 0],
            [0.5, S3H],
            [0.5, 0],
            [0.25, S3H/2],
            [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, S3H/6],
            [0.75, S3H/6],
            [0.5, 2*S3H/3],
            [0.5, S3H/2]
        ])
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts
