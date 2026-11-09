"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def perimeter_constraint(w, h):
    """Perimeter constraint: 2*(w+h) <= 4, i.e., w+h <= 2."""
    return 2.0 - (w + h)

def check_validity(circles):
    """Check if circles are valid: non-negative radii, disjoint interiors, within rectangle."""
    if circles is None or len(circles) != N:
        return False
    # Extract positions and radii
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Check radii are non-negative
    if np.any(rs < 0):
        return False
    
    # Check circles are within some rectangle of perimeter <= 4
    # We need to find if there exists a rectangle containing all circles with perimeter <= 4
    # For a given set of circles, the minimal bounding rectangle is determined by min/max x and y
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if 2 * (w + h) > 4 + 1e-6:
        return False
    
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-6:
                return False
    
    return True

def objective(x):
    """Objective: maximize sum of radii. We minimize negative sum of radii."""
    # x = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN, w, h]
    # Actually, let's use a simpler parameterization.
    # We'll fix the rectangle to be w x h where w + h = 2 (perimeter = 4).
    # Circles must be inside [0, w] x [0, h].
    
    # But the aspect ratio w/h is also a variable. Let's parameterize by aspect ratio.
    # Let w = 2 * t / (1 + t) and h = 2 / (1 + t) where t = w/h > 0.
    # Actually simpler: let w be a variable, h = 2 - w.
    
    # For now, let's just use the direct approach:
    # x = [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2 - w
    # Constraints:
    #   xi - ri >= 0, xi + ri <= w
    #   yi - ri >= 0, yi + ri <= h
    #   (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    n = N
    # Unpack
    coords = x[:3 * n]
    w = x[3 * n]
    h = 2.0 - w
    
    if w <= 0 or h <= 0:
        return 1e10
    
    xs = coords[0::3]
    ys = coords[1::3]
    rs = coords[2::3]
    
    # Penalty for constraint violations
    penalty = 0.0
    
    # Boundary constraints
    for i in range(n):
        if xs[i] - rs[i] < 0:
            penalty += (xs[i] - rs[i]) ** 2
        if xs[i] + rs[i] > w:
            penalty += (xs[i] + rs[i] - w) ** 2
        if ys[i] - rs[i] < 0:
            penalty += (ys[i] - rs[i]) ** 2
        if ys[i] + rs[i] > h:
            penalty += (ys[i] + rs[i] - h) ** 2
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist:
                penalty += (min_dist - np.sqrt(max(dist_sq, 0))) ** 2
    
    # Negative sum of radii (we minimize this)
    return -np.sum(rs) + 100.0 * penalty

def initial_guess(w, h):
    """Generate an initial guess for circle positions and radii."""
    # Grid-based initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = min(w, h) / (2 * side) * 0.95
            cells.append([cx, cy, r])
    
    # Pad if necessary
    while len(cells) < N:
        cells.append([w / 2, h / 2, min(w, h) / (2 * side) * 0.9])
    
    cells = cells[:N]
    x = np.array(cells).flatten()
    x = np.append(x, w)
    return x

def solve_packing(w_init):
    """Solve the packing problem for a given initial aspect ratio."""
    h_init = 2.0 - w_init
    x0 = initial_guess(w_init, h_init)
    
    bounds = []
    for i in range(N):
        bounds.append((0, w_init))  # xi
        bounds.append((0, h_init))  # yi
        bounds.append((0, min(w_init, h_init) / 2))  # ri
    
    # w is in (0, 2)
    bounds.append((0.1, 1.9))
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
    
    if result.success or result.fun < -0.1:
        x = result.x
        n = N
        coords = x[:3 * n]
        w = x[3 * n]
        h = 2.0 - w
        
        circles = np.zeros((N, 3))
        circles[:, 0] = coords[0::3]
        circles[:, 1] = coords[1::3]
        circles[:, 2] = coords[2::3]
        
        # Check validity
        if check_validity(circles):
            return circles, np.sum(circles[:, 2])
    
    return None, -np.inf

def construct():
    """Main entry point: construct 21 circles in a perimeter-4 rectangle."""
    best_circles = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    
    for ar in aspect_ratios:
        if time_left() < 10:
            break
        
        # w/h = ar, w + h = 2
        # w = ar * h, so h = 2 / (1 + ar), w = 2 * ar / (1 + ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        circles, score = solve_packing(w)
        
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles
    
    # If no valid solution found, fall back to a simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.9
                cells.append([cx, cy, r])
        best_circles = np.array(cells[:N])
    
    return best_circles
