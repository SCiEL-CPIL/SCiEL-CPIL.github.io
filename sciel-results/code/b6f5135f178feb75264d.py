"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _objective(params, n):
    """
    params: flat array of length 3n: [x0,y0,r0, x1,y1,r1, ...]
    We want to maximize sum(r_i) subject to:
      - circles inside [0,1]^2: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
      - circles disjoint: dist(i,j) >= r_i + r_j
    We convert to a minimization problem with penalty terms.
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        # x_i - r_i >= 0
        if x[i] < r[i]:
            penalty += (r[i] - x[i])**2 * 1000
        # 1 - x_i - r_i >= 0
        if 1.0 - x[i] < r[i]:
            penalty += (r[i] - (1.0 - x[i]))**2 * 1000
        # y_i - r_i >= 0
        if y[i] < r[i]:
            penalty += (r[i] - y[i])**2 * 1000
        # 1 - y_i - r_i >= 0
        if 1.0 - y[i] < r[i]:
            penalty += (r[i] - (1.0 - y[i]))**2 * 1000
    
    # Penalty for overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2 * 1000
    
    return obj + penalty

def _make_valid(circles):
    """
    Given an array of shape (n, 3) with columns (x, y, r),
    adjust radii to ensure validity:
    - Clamp positions and radii so circles are inside [0,1]^2
    - Iteratively shrink radii to resolve overlaps
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Clamp radii to be non-negative and reasonable
    r = np.clip(r, 1e-6, 0.5)
    
    # Clamp positions to be inside the square given the radius
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    
    # Iteratively resolve overlaps by shrinking radii
    for iteration in range(1000):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Need to shrink radii so that r[i] + r[j] <= dist
                    # Split the deficit equally
                    deficit = min_dist - dist
                    # Shrink each by deficit/2, but not below 1e-6
                    new_ri = max(1e-6, r[i] - deficit/2.0)
                    new_rj = max(1e-6, r[j] - deficit/2.0)
                    if new_ri < r[i]:
                        r[i] = new_ri
                        changed = True
                    if new_rj < r[j]:
                        r[j] = new_rj
                        changed = True
                    # Re-clamp positions
                    x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                    y[i] = np.clip(y[i], r[i], 1.0 - r[i])
                    x[j] = np.clip(x[j], r[j], 1.0 - r[j])
                    y[j] = np.clip(y[j], r[j], 1.0 - r[j])
        if not changed:
            break
    
    return np.column_stack([x, y, r])

def _maximize_radii(circles, max_iter=200):
    """
    Given valid circles, try to increase radii while maintaining validity.
    Uses a simple iterative approach: try to increase each radius slightly,
    and if it causes overlap, shrink the problematic pairs.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Try to increase radii by checking if we can grow them
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            # Try to increase r[i] by a small amount
            delta = 1e-4
            new_r = r[i] + delta
            # Check boundary
            if new_r > 1.0 - max(x[i], y[i]) or new_r > min(x[i], y[i]):
                continue
            # Check overlaps
            valid = True
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < new_r + r[j]:
                    valid = False
                    break
            if valid:
                r[i] = new_r
                improved = True
        if not improved:
            break
    
    return np.column_stack([x, y, r])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial grid-based seed
    side = int(math.ceil(math.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 1e-3
            cells.append([cx, cy, cr])
    initial = np.array(cells[:N])
    
    best_circles = initial.copy()
    best_score = np.sum(best_circles[:, 2])
    
    # Multi-start optimization
    np.random.seed(42)
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if restart == 0:
            params0 = best_circles.flatten()
        else:
            # Random initial positions
            x0 = np.random.uniform(0.1, 0.9, N)
            y0 = np.random.uniform(0.1, 0.9, N)
            r0 = np.full(N, 0.15)  # Start with moderate radii
            params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 1.0))  # x
            bounds.append((1e-6, 1.0))  # y
            bounds.append((1e-6, 0.5))  # r
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < 0:  # Check if we got a reasonable result
                x = result.x[:N]
                y = result.x[N:2*N]
                r = result.x[2*N:]
                circles = np.column_stack([x, y, r])
                
                # Make valid and maximize radii
                circles = _make_valid(circles)
                circles = _maximize_radii(circles)
                
                score = np.sum(circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
        except Exception:
            pass
    
    # Final validation
    best_circles = _make_valid(best_circles)
    return best_circles
# EVOLVE-BLOCK-END
