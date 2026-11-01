"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START

def _rect_perimeter(x, y, r):
    """Perimeter of bounding rectangle: 2*(x_max - x_min + y_max - y_min)."""
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    return 2.0 * ((x_max - x_min) + (y_max - y_min))

def _make_objective(r):
    """Objective: minimize sum of radii (we want to maximize)."""
    return -np.sum(r)

def _make_constraints(x, y):
    """
    Constraints:
    1. Perimeter <= 4
    2. Circle i and j do not overlap: dist >= r_i + r_j
    """
    n = len(x)
    
    def perimeter_constraint(v):
        r = v
        p = _rect_perimeter(x, y, r)
        return 4.0 - p  # >= 0
    
    # Non-overlap constraints
    def overlap_constraint(v, i, j):
        r = v
        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
        return dist - (r[i] + r[j])
    
    # We'll handle this via penalty or direct constraint
    # For scipy, let's use a penalty approach within the objective
    # Or use SLSQP with many constraints
    
    # Actually, let's reformulate: use a penalty method
    # Objective: -sum(r) + lambda * max(0, perimeter - 4) + mu * sum(max(0, r_i + r_j - dist_ij)^2)
    
    pass

def _solve_with_penalty(x, y, r_init, max_iter=5000, tol=1e-8):
    """
    Solve using penalty method.
    Variables: r (radii)
    Objective: minimize -sum(r) + penalty terms
    """
    n = len(r_init)
    r = np.array(r_init, dtype=float)
    
    # Precompute distances between centers
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            dists[i,j] = d
            dists[j,i] = d
    
    def objective(v):
        r = v
        # Base objective: maximize sum of radii
        obj = -np.sum(r)
        
        # Perimeter penalty
        p = _rect_perimeter(x, y, r)
        if p > 4.0:
            obj += 100.0 * (p - 4.0)**2
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                overlap = r[i] + r[j] - dists[i,j]
                if overlap > 0:
                    obj += 100.0 * overlap**2
        
        # Radius positivity
        for i in range(n):
            if r[i] < 0:
                obj += 100.0 * r[i]**2
        
        return obj
    
    def jac(v):
        r = v
        grad = -np.ones(n)
        
        # Perimeter gradient
        p = _rect_perimeter(x, y, r)
        if p > 4.0:
            # d(perimeter)/d(r_i)
            # perimeter = 2*(max(x+r) - min(x-r) + max(y+r) - min(y-r))
            # d/dr_i of max(x+r) is 1 if i is the max, else 0
            # d/dr_i of min(x-r) is -1 if i is the min, else 0
            x_plus = x + r
            x_minus = x - r
            y_plus = y + r
            y_minus = y - r
            
            max_x_idx = np.argmax(x_plus)
            min_x_idx = np.argmin(x_minus)
            max_y_idx = np.argmax(y_plus)
            min_y_idx = np.argmin(y_minus)
            
            for i in range(n):
                dp = 0.0
                if i == max_x_idx:
                    dp += 1.0
                if i == min_x_idx:
                    dp -= 1.0
                if i == max_y_idx:
                    dp += 1.0
                if i == min_y_idx:
                    dp -= 1.0
                grad[i] += 200.0 * (p - 4.0) * dp
        
        # Overlap gradient
        for i in range(n):
            for j in range(i+1, n):
                overlap = r[i] + r[j] - dists[i,j]
                if overlap > 0:
                    grad[i] += 200.0 * overlap
                    grad[j] += 200.0 * overlap
        
        # Positivity gradient
        for i in range(n):
            if r[i] < 0:
                grad[i] += 200.0 * r[i]
        
        return grad
    
    # Use L-BFGS-B with bounds
    bounds = [(0.0, None)] * n
    
    best_r = r.copy()
    best_obj = objective(r)
    
    for iteration in range(max_iter):
        res = minimize(objective, r, jac=jac, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
        
        r_new = res.x
        obj_new = objective(r_new)
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_r = r_new.copy()
        
        if np.max(np.abs(r_new - r)) < tol:
            break
        r = r_new
    
    return best_r

def _find_optimal_layout():
    """
    Find optimal layout by optimizing both positions and radii.
    This is a non-convex problem, so we use multiple restarts.
    """
    deadline = time.time() + 140  # 140s deadline (10s margin)
    
    best_total_r = 0.0
    best_result = None
    
    # Initial layouts to try
    layouts = []
    
    # Grid layout
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(layouts) < N:
                layouts.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Hexagonal-like packing
    # Try different aspect ratios
    for aspect in [0.5, 0.66, 0.75, 1.0, 1.33, 1.5, 2.0]:
        w = 2.0 * aspect  # width
        h = 2.0 / aspect  # height
        # Place circles in a grid within this rectangle
        cols = max(1, int(np.sqrt(N * w / h)))
        rows = max(1, int(np.ceil(N / cols)))
        for k in range(N):
            i = k % cols
            j = k // cols
            x = (i + 0.5) / cols * w
            y = (j + 0.5) / rows * h
            layouts.append((x, y))
    
    # Random layouts
    np.random.seed(42)
    for _ in range(20):
        x = np.random.uniform(0, 2, N)
        y = np.random.uniform(0, 2, N)
        for i in range(N):
            layouts.append((x[i], y[i]))
    
    best_layout_x = None
    best_layout_y = None
    best_layout_obj = float('inf')
    
    for layout in layouts:
        if time.time() > deadline:
            break
        
        x = np.array([p[0] for p in layout[:N]])
        y = np.array([p[1] for p in layout[:N]])
        
        # Initial radii: small
        r_init = np.full(N, 0.05)
        
        r = _solve_with_penalty(x, y, r_init, max_iter=2000, tol=1e-9)
        
        # Check validity
        p = _rect_perimeter(x, y, r)
        valid = True
        if p > 4.0 + 1e-6:
            valid = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-6:
                    valid = False
        
        if valid:
            total_r = np.sum(r)
            if total_r > best_total_r:
                best_total_r = total_r
                best_result = np.column_stack([x, y, r])
        
        # Track best layout for joint optimization
        obj = _rect_perimeter(x, y, r)
        if obj < best_layout_obj and p <= 4.0 + 1e-6:
            best_layout_obj = obj
            best_layout_x = x.copy()
            best_layout_y = y.copy()
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid but possibly suboptimal solution
    # Use the grid layout with adjusted radii
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for k in range(N):
        i = k % side
        j = k // side
        x[k] = (i + 0.5) / side
        y[k] = (j + 0.5) / side
        r[k] = 0.5 / side - 1e-3
    
    return np.column_stack([x, y, r])

def construct():
    """Construct 21 disjoint circles in a rectangle of perimeter at most 4."""
    result = _find_optimal_layout()
    
    # Verify and clean up
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    
    # Ensure non-negative radii
    r = np.maximum(r, 0.0)
    
    # If perimeter is slightly over, scale down
    p = _rect_perimeter(x, y, r)
    if p > 4.0:
        scale = 4.0 / p
        x = x * scale
        y = y * scale
        r = r * scale
    
    return result

# EVOLVE-BLOCK-END
