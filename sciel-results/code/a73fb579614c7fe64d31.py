"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _check_validity(circles, tol=1e-9):
    """Check if circles are valid: disjoint interiors, bounding box perimeter <= 4."""
    n = len(circles)
    if n == 0:
        return True
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii positive
    if np.any(rs <= 0):
        return False
    
    # Check bounding box
    x_min = xs - rs
    x_max = xs + rs
    y_min = ys - rs
    y_max = ys + rs
    
    W = x_max - x_min
    H = y_max - y_min
    
    if W < 0 or H < 0:
        return False
    
    perimeter = 2 * (W + H)
    if perimeter > 4.0 + tol:
        return False
    
    # Check disjoint interiors: distance between centers >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True

def _objective(x, n, W, H):
    """
    Objective: maximize sum of radii.
    x layout: [x0, y0, r0, x1, y1, r1, ...]
    We minimize -sum(r).
    """
    rs = x[2::3]
    return -np.sum(rs)

def _constraints(x, n, W, H):
    """
    Constraints:
    1. Each circle inside rectangle [0, W] x [0, H]:
       r_i <= x_i, r_i <= W - x_i, r_i <= y_i, r_i <= H - y_i
    2. Pairwise disjoint: dist_ij >= r_i + r_j
       Equivalent to: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
       For optimization, we use: r_i + r_j - dist_ij <= 0
       But this is non-smooth. We can use a penalty or smooth approximation.
       Alternatively, we can use: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
       This is a quadratic constraint.
    """
    cons = []
    
    # Boundary constraints: r_i <= x_i, etc.
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        # r_i - x_i <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        # r_i - (W - x_i) <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, W=W: (W - x[3*i]) - x[3*i+2]})
        # r_i - y_i <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        # r_i - (H - y_i) <= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, H=H: (H - x[3*i+1]) - x[3*i+2]})
    
    # Pairwise constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # In inequality form for minimize: -[(x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2] <= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*j] - x[3*i])**2 + (x[3*j+1] - x[3*i+1])**2 - (x[3*i+2] + x[3*j+2])**2
            })
    
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We need to find W, H such that 2(W+H) <= 4, i.e., W+H <= 2.
    # Let's try a few aspect ratios. For each, solve the optimization.
    # Start with a grid-like initial guess.
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # W + H = 2. Let H = 2 - W. W in (0, 2).
    # Symmetric around W=1. Try W in [0.5, 1.5].
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.666, 1.5, 0.75, 1.333]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
        
        W = aspect / (1 + aspect) * 2  # W + H = 2, H = 2 - W
        H = 2.0 - W
        
        # Initial guess: grid layout
        side = int(np.ceil(np.sqrt(N)))
        # Place circles in a grid within [0, W] x [0, H]
        x0 = np.zeros(3 * N)
        for i in range(N):
            row = i // side
            col = i % side
            # Center in cell
            cx = (col + 0.5) * W / side
            cy = (row + 0.5) * H / side
            r = min(W / (2 * side), H / (2 * side)) * 0.9
            x0[3*i] = cx
            x0[3*i+1] = cy
            x0[3*i+2] = r
        
        # Add small random perturbation
        x0 += np.random.randn(3 * N) * 0.01
        # Ensure radii positive
        x0[2::3] = np.abs(x0[2::3]) + 0.01
        
        # Bounds: x, y in [0, W] or [0, H], r >= 0
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((0, min(W, H) / 2))
        
        cons = _constraints(x0, N, W, H)
        
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < best_value:
                circles = res.x.reshape(N, 3)
                # Validate
                if _check_validity(circles):
                    val = np.sum(circles[:, 2])
                    if val > best_value:
                        best_value = val
                        best_result = circles
        except Exception as e:
            pass
    
    # Also try a simpler approach: equal circles in a grid
    # For 21 circles, try 5x5 grid with one removed, or 7x3, etc.
    for side in range(3, 8):
        if time.time() > deadline - 5.0:
            break
        
        # Try W = H = 1 (square)
        W = 1.0
        H = 1.0
        # But perimeter = 4, so W+H=2, W=H=1 is valid.
        
        # Grid of side x side
        if side * side < N:
            continue
        
        x0 = np.zeros(3 * N)
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r = 0.5 / side * 0.95
            x0[3*i] = cx
            x0[3*i+1] = cy
            x0[3*i+2] = r
        
        circles = x0.reshape(N, 3)
        if _check_validity(circles):
            val = np.sum(circles[:, 2])
            if val > best_value:
                best_value = val
                best_result = circles
    
    # If we have a valid result, return it. Otherwise, return a trivial valid one.
    if best_result is not None:
        return best_result
    
    # Fallback: very small circles in a grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.1 / side) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
