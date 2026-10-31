"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145  # safety margin before 150s kill

def _time_ok():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def _rectangle_dims(w):
    """Given width w, height h = 2 - w (perimeter = 4)."""
    return w, 2.0 - w

def _initial_hex_config(w, h, n_circles):
    """Generate initial hexagonal-like positions in rectangle [0,w]x[0,h]."""
    # Estimate number of rows and columns
    # Hexagonal packing: row offset
    # Try to fit n_circles in a hex pattern
    best_positions = None
    best_score = -np.inf
    
    for n_cols in range(1, n_circles + 1):
        n_rows = (n_circles + n_cols - 1) // n_cols
        if n_rows == 0:
            continue
        positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                if len(positions) >= n_circles:
                    break
                # Hex offset: odd rows shifted
                x_base = (col + 0.5) / n_cols
                if row % 2 == 1:
                    x_base = x_base + 0.5 / n_cols
                x_base = min(max(x_base, 0), 1)
                y_base = (row + 0.5) / n_rows
                x = x_base * w
                y = y_base * h
                positions.append([x, y])
        
        if len(positions) != n_circles:
            continue
        
        positions = np.array(positions)
        # Score: sum of min distances to neighbors (rough measure of spread)
        if len(positions) > 1:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            score = np.sum(min_dists)
        else:
            score = 0.0
        
        if score > best_score:
            best_score = score
            best_positions = positions
    
    if best_positions is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n_circles)))
        positions = []
        for i in range(n_circles):
            row = i // side
            col = i % side
            x = (col + 0.5) / side * w
            y = (row + 0.5) / side * h
            positions.append([x, y])
        best_positions = np.array(positions)
    
    return best_positions

def _evaluate_and_refine(w, h, positions0, max_iter=2000, lr=0.01):
    """
    Optimize circle positions and radii for given rectangle.
    Uses gradient-free local search with penalty method.
    Returns (positions, radii, sum_radii).
    """
    n = len(positions0)
    # State: [x0, y0, r0, x1, y1, r1, ...]
    # Initial radii: small
    r0 = np.full(n, 0.02)
    state0 = np.concatenate([positions0.flatten(), r0])
    
    def objective(state):
        pos = state[:2*n].reshape(n, 2)
        rad = state[2*n:]
        
        # Penalty for out of bounds
        penalty = 0.0
        for i in range(n):
            x, y = pos[i]
            r = rad[i]
            if x - r < 0: penalty += (x - r) ** 2 * 100
            if x + r > w: penalty += (x + r - w) ** 2 * 100
            if y - r < 0: penalty += (y - r) ** 2 * 100
            if y + r > h: penalty += (y + r - h) ** 2 * 100
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100
        
        # Maximize sum of radii, so minimize negative sum
        return -np.sum(rad) + penalty
    
    # Use Nelder-Mead with restarts
    best_state = state0.copy()
    best_obj = objective(state0)
    
    # Simple local search: random perturbations
    np.random.seed(42)
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Random perturbation
        perturbation = np.random.randn(len(state0)) * lr
        new_state = best_state + perturbation
        
        # Ensure radii are positive
        new_state[2*n:] = np.maximum(new_state[2*n:], 0.001)
        
        new_obj = objective(new_state)
        if new_obj < best_obj:
            best_state = new_state
            best_obj = new_obj
            # Increase lr slightly
            lr = min(lr * 1.05, 0.05)
        else:
            lr = max(lr * 0.95, 0.001)
    
    pos = best_state[:2*n].reshape(n, 2)
    rad = best_state[2*n:]
    
    # Final cleanup: clamp to valid bounds
    for i in range(n):
        rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
        rad[i] = max(rad[i], 0.0)
    
    # Check for overlaps and shrink if needed
    for it in range(50):
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rad[i] + rad[j]:
                    # Shrink both
                    shrink = (rad[i] + rad[j] - dist) / 2 + 0.001
                    rad[i] = max(rad[i] - shrink, 0.0)
                    rad[j] = max(rad[j] - shrink, 0.0)
                    overlap_found = True
        # Check bounds
        for i in range(n):
            rad[i] = min(rad[i], pos[i,0], w - pos[i,0], pos[i,1], h - pos[i,1])
            rad[i] = max(rad[i], 0.0)
        if not overlap_found:
            break
    
    return pos, rad, np.sum(rad)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_result = None
    best_sum = -np.inf
    
    # Try different aspect ratios
    # w + h = 2, so w in (0, 2), h = 2 - w
    # For 21 circles, aspect ratio around 1 to 2 might be good
    aspect_ratios = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 0.8, 0.9]
    
    for ar in aspect_ratios:
        if not _time_ok():
            break
        
        # w/h = ar, w + h = 2
        # w = ar * h, so ar*h + h = 2, h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1)
        w = ar * h
        
        # Generate initial configuration
        positions0 = _initial_hex_config(w, h, N)
        
        # Optimize
        pos, rad, total = _evaluate_and_refine(w, h, positions0, max_iter=1500)
        
        if total > best_sum:
            best_sum = total
            best_result = np.column_stack([pos, rad])
    
    # If we didn't find anything, use fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Final validation: ensure no overlaps and within bounds
    # Get the rectangle that was used
    # Actually, we need to return circles that fit in SOME rectangle with perimeter <= 4
    # The evaluator will check the bounding box of the circles
    
    # Let's validate and fix
    pos = best_result[:, :2]
    rad = best_result[:, 2]
    
    # Compute bounding box
    x_min = np.min(pos[:, 0] - rad)
    x_max = np.max(pos[:, 0] + rad)
    y_min = np.min(pos[:, 1] - rad)
    y_max = np.max(pos[:, 1] + rad)
    
    rect_w = x_max - x_min
    rect_h = y_max - y_min
    perimeter = 2 * (rect_w + rect_h)
    
    # If perimeter > 4, scale down
    if perimeter > 4:
        scale = 4.0 / perimeter
        pos = pos * scale
        rad = rad * scale
        # Recompute bounding box
        x_min = np.min(pos[:, 0] - rad)
        x_max = np.max(pos[:, 0] + rad)
        y_min = np.min(pos[:, 1] - rad)
        y_max = np.max(pos[:, 1] + rad)
        rect_w = x_max - x_min
        rect_h = y_max - y_min
        perimeter = 2 * (rect_w + rect_h)
    
    # Ensure non-negative radii
    rad = np.maximum(rad, 0.0)
    
    return np.column_stack([pos, rad])
