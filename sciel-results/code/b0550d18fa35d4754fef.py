"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _validate(w, h, circles):
    """Check if circles are valid: within bounds and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0 or cx - r < -1e-8 or cx + r > w + 1e-8 or cy - r < -1e-8 or cy + r > h + 1e-8:
            return False
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - 1e-8:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(w, h, n_circles, staggered=True):
    """Create a staggered grid initial placement."""
    circles = []
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles * h / w)))
    cols = int(np.ceil(n_circles / rows))
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            if staggered and i % 2 == 1:
                cx = (j + 0.5) / cols * w
            else:
                cx = (j + 0.25) / cols * w
            cy = (i + 0.5) / rows * h
            # Estimate radius based on local spacing
            dx = w / cols
            dy = h / rows
            r = 0.45 * min(dx, dy)
            # Clamp to stay within bounds
            r = min(r, cx, w - cx, cy, h - cy)
            if r > 0:
                circles.append((cx, cy, r))
    
    # If we don't have enough, add more with smaller radii
    while len(circles) < n_circles:
        idx = len(circles)
        cx = (idx + 0.5) / n_circles * w
        cy = 0.5 * h
        r = 0.02
        r = min(r, cx, w - cx, cy, h - cy)
        if r > 0:
            circles.append((cx, cy, r))
    
    return circles[:n_circles]

def _local_optimize(w, h, init_circles, deadline, penalty_weight=100.0):
    """Local optimization with penalty-based constraints."""
    x0 = np.array([v for c in init_circles for v in c])
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        circles = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(N)]
        
        # Penalty for boundary violations
        penalty = 0.0
        for cx, cy, r in circles:
            if r <= 0:
                penalty += penalty_weight * (1.0 - r)
            if cx - r < 0:
                penalty += penalty_weight * (cx - r)
            if cx + r > w:
                penalty += penalty_weight * (cx + r - w)
            if cy - r < 0:
                penalty += penalty_weight * (cy - r)
            if cy + r > h:
                penalty += penalty_weight * (cy + r - h)
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist:
                    penalty += penalty_weight * (min_dist - dist)
        
        return -_total_radius(circles) + penalty
    
    # Bounds
    bounds = []
    for _ in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, 0.5))
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < 0:  # Valid solution (negative means total radius minus penalty)
            circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(N)]
            if _validate(w, h, circles):
                return circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75, 1.2, 0.8]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 / (1 + aspect)
        
        # Grid initialization
        init_circles = _grid_init(w, h, N, staggered=True)
        
        # Local optimization
        result = _local_optimize(w, h, init_circles, deadline)
        if result is not None:
            score = _total_radius(result)
            if score > best_score:
                best_score = score
                best_circles = result
        
        # Also try with different stagger patterns
        if time.time() < deadline - 30:
            init_circles2 = _grid_init(w, h, N, staggered=False)
            result2 = _local_optimize(w, h, init_circles2, deadline)
            if result2 is not None:
                score2 = _total_radius(result2)
                if score2 > best_score:
                    best_score = score2
                    best_circles = result2
    
    # Random restarts with local optimization
    if time.time() < deadline - 20:
        w, h = 1.0, 1.0  # Use square for random restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random initial positions
            random_circles = []
            for _ in range(N):
                cx = np.random.uniform(0.1, 0.9)
                cy = np.random.uniform(0.1, 0.9)
                r = np.random.uniform(0.02, 0.08)
                random_circles.append((cx, cy, r))
            
            result = _local_optimize(w, h, random_circles, deadline)
            if result is not None:
                score = _total_radius(result)
                if score > best_score:
                    best_score = score
                    best_circles = result
    
    # Fallback: simple grid if nothing else worked
    if best_circles is None:
        w, h = 1.0, 1.0
        best_circles = _grid_init(w, h, N, staggered=True)
    
    return np.array(best_circles)
# EVOLVE-BLOCK-END
