"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import sys

N = 21
PERIMETER = 4.0
MAX_WALL = 150.0  # evaluator limit; we target well below

def _pack_for_aspect(w, h, max_iters=800, tol=1e-7):
    """
    Pack N circles in rectangle [0,w]x[0,h], maximize sum of radii.
    Uses scipy minimize with linear constraints for non-negativity and
    nonlinear constraints for non-overlap (squared distance >= (ri+rj)^2).
    Actually, non-overlap is non-linear (quadratic), so we use SLSQP.
    """
    # Initial guess: grid layout
    cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0 else N
    cols = max(1, min(N, cols))
    rows = int(np.ceil(N / cols))
    
    # Build initial positions
    centers = np.zeros((N, 2))
    for i in range(N):
        r_idx = i // cols
        c_idx = i % cols
        centers[i, 0] = (c_idx + 0.5) * w / cols
        centers[i, 1] = (r_idx + 0.5) * h / rows
    
    # Initial radii: uniform small value
    r_init = 0.01 * np.ones(N)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = r_init[i]
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    # Inequality constraints: non-overlap and boundary
    # For SLSQP, constraints are of the form f(v) >= 0
    def make_nonoverlap(i, j):
        def constraint(v):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            ri = v[3*i+2]
            rj = v[3*j+2]
            return dx*dx + dy*dy - (ri + rj)**2
        constraint.name = f"nonoverlap_{i}_{j}"
        return constraint
    
    def make_boundary_x(i):
        def constraint(v):
            return np.minimum(v[3*i] - v[3*i+2], w - v[3*i] - v[3*i+2])
        constraint.name = f"boundary_x_{i}"
        return constraint
    
    def make_boundary_y(i):
        def constraint(v):
            return np.minimum(v[3*i+1] - v[3*i+2], h - v[3*i+1] - v[3*i+2])
        constraint.name = f"boundary_y_{i}"
        return constraint
    
    def make_radius(i):
        def constraint(v):
            return v[3*i+2]
        constraint.name = f"radius_{i}"
        return constraint
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': make_boundary_x(i)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_y(i)})
        constraints.append({'type': 'ineq', 'fun': make_radius(i)})
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_nonoverlap(i, j)})
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, 0.5 * min(w, h)))
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': max_iters, 'ftol': tol})
    
    if result.success:
        return result.x
    return x0

def _validate_and_score(positions, w, h):
    """Validate packing and compute sum of radii."""
    N = len(positions) // 3
    x = positions[0::3]
    y = positions[1::3]
    r = positions[2::3]
    
    # Check non-negativity
    if np.any(r <= 0):
        return 0.0
    
    # Check boundary
    if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
        return 0.0
    
    # Check non-overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)

def construct():
    start_time = time.time()
    best_value = 0.0
    best_positions = None
    best_w = None
    best_h = None
    
    # Try various aspect ratios
    aspect_ratios = []
    for ar in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        # Perimeter = 2*(w+h) = 4, so w+h = 2
        # w/h = ar => w = ar*h => h*(ar+1) = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        aspect_ratios.append((w, h))
    
    # Add some random perturbations around 1:1
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = 2.0 / (1.0 + rng.uniform(0.8, 1.2))
        w = 2.0 - h
        aspect_ratios.append((w, h))
    
    for w, h in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > MAX_WALL - 30:  # safety margin
            break
        
        try:
            positions = _pack_for_aspect(w, h, max_iters=500, tol=1e-6)
            value = _validate_and_score(positions, w, h)
            if value > best_value:
                best_value = value
                best_positions = positions
                best_w = w
                best_h = h
        except Exception as e:
            continue
    
    if best_positions is None:
        # Fallback: simple grid in 2x2 rectangle
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        return np.array(cells)
    
    # Construct output: [x, y, r] for each circle
    x = best_positions[0::3]
    y = best_positions[1::3]
    r = best_positions[2::3]
    
    result = np.column_stack([x, y, r])
    return result
