"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 21
DEADLINE = 115.0  # Safety margin before 120s limit (with some buffer)
START_TIME = time.time()

def check_time():
    """Check if we're close to the deadline."""
    return (time.time() - START_TIME) > DEADLINE

def rect_perimeter(w, h):
    return 2.0 * (w + h)

def validate_and_score(circles):
    """
    Validate circles and compute score.
    circles: (n, 3) array of [x, y, r]
    Returns (is_valid, score, w, h)
    """
    if len(circles) < N:
        return False, 0.0, 0, 0
    
    # Get bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return False, 0.0, 0, 0
    
    # Compute bounding rectangle
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w <= 0 or h <= 0:
        return False, 0.0, 0, 0
    
    p = 2.0 * (w + h)
    if p > 4.0 + 1e-9:
        return False, 0.0, w, h
    
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False, 0.0, w, h
    
    score = np.sum(rs)
    return True, score, w, h

def make_grid_layout(n_circles, w, h):
    """Create a grid-based layout of circles in a w x h rectangle."""
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n_circles * w / h)))
    rows = int(np.ceil(n_circles / cols))
    
    if rows < 1:
        rows = 1
    if cols < 1:
        cols = 1
    
    # Adjust to fit
    while rows * cols < n_circles:
        if (rows + 1) * cols < cols * rows:
            rows += 1
        else:
            cols += 1
    
    # Place circles in grid
    positions = []
    radii = []
    
    # Cell dimensions
    cell_w = w / cols
    cell_h = h / rows
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n_circles:
                break
            cx = j * cell_w + cell_w / 2
            cy = i * cell_h + cell_h / 2
            # Radius is limited by cell size and neighbors
            r = min(cell_w, cell_h) / 2.0 * 0.95  # Small margin
            positions.append((cx, cy))
            radii.append(r)
    
    # If we have too few, add more in remaining space
    while len(positions) < n_circles:
        # Add at random positions in the rectangle
        cx = np.random.uniform(0, w)
        cy = np.random.uniform(0, h)
        r = 0.05
        positions.append((cx, cy))
        radii.append(r)
    
    return np.array([(positions[i][0], positions[i][1], radii[i]) for i in range(n_circles)])

def smooth_min(values, k=100.0):
    """Smooth minimum using log-sum-exp trick."""
    values = np.array(values)
    min_val = np.min(values)
    return min_val - np.log(np.sum(np.exp(k * (values - min_val)))) / k

def objective(params, w, h):
    """
    Objective function for optimization.
    params: [x1, y1, r1, x2, y2, r2, ...] of length 3*N
    Returns negative sum of radii (to minimize) + penalty for constraint violations.
    """
    params = np.array(params)
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Penalty for negative radii
    pen_neg_r = np.sum(np.maximum(0, -rs)) ** 2
    
    # Penalty for circles going outside bounding box
    # The bounding box is defined by the circles themselves, so this is tricky.
    # Instead, we'll work with a fixed aspect ratio and optimize within that.
    
    # Compute actual bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w_actual = max_x - min_x
    h_actual = max_y - min_y
    
    if w_actual <= 0 or h_actual <= 0:
        return 1e10
    
    p_actual = 2.0 * (w_actual + h_actual)
    
    # Penalty for perimeter exceeding 4
    pen_perim = np.maximum(0, p_actual - 4.0) ** 2
    
    # Penalty for overlapping circles
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                pen_overlap += overlap ** 2
    
    # Also add smooth repulsion for near-overlapping circles
    repulsion = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-10
            # Soft repulsion when circles are close
            if dist < rs[i] + rs[j] + 0.1:
                gap = dist - rs[i] - rs[j]
                if gap < 0.1:
                    repulsion += np.exp(-10 * gap) if gap > 0 else np.exp(10 * gap)
    
    total_penalty = pen_neg_r * 100.0 + pen_perim * 1000.0 + pen_overlap * 100.0 + repulsion * 10.0
    
    # We want to maximize sum of radii, so minimize negative
    return -np.sum(rs) + total_penalty

def optimize_from_initial(initial_circles, max_iters=50, time_limit=None):
    """
    Optimize circle positions and radii starting from initial layout.
    """
    if time_limit is None:
        time_limit = 30
    
    n_params = 3 * N
    
    # Initial parameters
    x0 = np.zeros(n_params)
    for i in range(N):
        x0[3*i] = initial_circles[i, 0]
        x0[3*i+1] = initial_circles[i, 1]
        x0[3*i+2] = initial_circles[i, 2]
    
    # Bounds: radii must be positive, positions can be anything
    bounds = []
    for i in range(N):
        bounds.append((None, None))  # x
        bounds.append((None, None))  # y
        bounds.append((1e-6, None))  # r (positive)
    
    best_params = x0.copy()
    best_obj = objective(x0, 0, 0)
    
    start_opt = time.time()
    
    for it in range(max_iters):
        if (time.time() - start_opt) > time_limit:
            break
        
        # Try L-BFGS-B
        try:
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Small perturbation to escape local minima
        if it % 5 == 0 and it > 0:
            perturbation = np.random.normal(0, 0.01, size=n_params)
            perturbation[2::3] = np.abs(perturbation[2::3]) * 0.1  # Keep radii positive-ish
            test_params = best_params + perturbation
            test_params[2::3] = np.maximum(test_params[2::3], 1e-6)
            test_obj = objective(test_params, 0, 0)
            if test_obj < best_obj:
                best_params = test_params
                best_obj = test_obj
    
    # Convert back to circles
    circles = np.zeros((N, 3))
    for i in range(N):
        circles[i, 0] = best_params[3*i]
        circles[i, 1] = best_params[3*i+1]
        circles[i, 2] = max(best_params[3*i+2], 1e-6)
    
    return circles, best_obj

def polish(circles, time_limit=5.0):
    """
    High-precision polish pass: local search to refine positions and radii.
    """
    start_polish = time.time()
    circles = circles.copy()
    rs = circles[:, 2].copy()
    
    # First, check validity and get current perimeter
    valid, score, w, h = validate_and_score(circles)
    
    # Local search: try small perturbations
    best_score = score if valid else 0.0
    best_circles = circles.copy()
    
    n_steps = 0
    while (time.time() - start_polish) < time_limit:
        n_steps += 1
        
        # Try random small moves
        improved = False
        for _ in range(20):
            if (time.time() - start_polish) > time_limit:
                break
            
            trial = best_circles.copy()
            
            # Pick a random circle to move
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.02)
            dy = np.random.normal(0, 0.02)
            trial[idx, 0] += dx
            trial[idx, 1] += dy
            
            # Try to increase radius slightly
            if np.random.random() < 0.3:
                trial[idx, 2] *= (1.0 + np.random.uniform(0, 0.01))
            
            valid_t, score_t, w_t, h_t = validate_and_score(trial)
            if valid_t and score_t > best_score:
                best_score = score_t
                best_circles = trial
                improved = True
        
        # If no improvement after several attempts, try a different strategy
        if not improved and n_steps % 10 == 0:
            # Try expanding all radii slightly
            trial = best_circles.copy()
            trial[:, 2] *= 1.001
            valid_t, score_t, w_t, h_t = validate_and_score(trial)
            if valid_t and score_t > best_score:
                best_score = score_t
                best_circles = trial
    
    return best_circles

def construct():
    """
    Main entry point: pack 21 circles in a perimeter-4 rectangle.
    Returns array of shape (N, 3) with [x, y, r] for each circle.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Initial aspect ratio: start with square-ish
    # Perimeter 4 => w + h = 2
    # Start with w = h = 1 (square)
    
    best_circles = None
    best_score = 0.0
    
    # Generate multiple initial layouts with different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.3, 0.77]
    
    for ar in aspect_ratios:
        if check_time():
            break
        
        # w + h = 2, w/h = ar => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Create initial grid layout
        initial = make_grid_layout(N, w, h)
        
        # Optimize
        time_remaining = DEADLINE - (time.time() - START_TIME)
        opt_time = min(15.0, time_remaining * 0.5)
        
        if opt_time < 2.0:
            break
        
        optimized, obj = optimize_from_initial(initial, max_iters=20, time_limit=opt_time)
        
        # Validate
        valid, score, w_actual, h_actual = validate_and_score(optimized)
        
        if valid and score > best_score:
            best_score = score
            best_circles = optimized.copy()
        
        # Polish
        if best_circles is not None:
            time_remaining = DEADLINE - (time.time() - START_TIME)
            polish_time = min(5.0, time_remaining * 0.3)
            
            if polish_time > 1.0:
                polished = polish(best_circles, time_limit=polish_time)
                valid_p, score_p, w_p, h_p = validate_and_score(polished)
                if valid_p and score_p > best_score:
                    best_score = score_p
                    best_circles = polished
    
    # If we still don't have a valid solution, create a simple fallback
    if best_circles is None or not validate_and_score(best_circles)[0]:
        # Simple fallback: grid of 21 circles in a 1x1 square
        # 5x5 grid minus 4 corners = 21
        positions = []
        radii = []
        for i in range(5):
            for j in range(5):
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                cx = (j + 0.5) * 0.2
                cy = (i + 0.5) * 0.2
                r = 0.09
                positions.append((cx, cy))
                radii.append(r)
        
        best_circles = np.array([(positions[i][0], positions[i][1], radii[i]) for i in range(min(N, len(positions)))])
        
        # Pad if needed
        while len(best_circles) < N:
            best_circles = np.vstack([best_circles, [[0.5, 0.5, 0.01]]])
    
    # Final validation and return
    valid, score, w, h = validate_and_score(best_circles)
    
    # Ensure we return exactly N circles
    if len(best_circles) > N:
        best_circles = best_circles[:N]
    elif len(best_circles) < N:
        # This shouldn't happen, but handle it
        while len(best_circles) < N:
            best_circles = np.vstack([best_circles, [[0.5, 0.5, 0.01]]])
    
    return best_circles
