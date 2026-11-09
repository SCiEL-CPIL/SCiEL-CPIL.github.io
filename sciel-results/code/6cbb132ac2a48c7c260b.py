import numpy as np
import time
from scipy.optimize import minimize
import itertools

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Leave 10s margin
    
    # Initial guess: 3x3 grid
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    sides = np.full(n, side, dtype=np.float64)
    
    best_score = np.sum(sides)
    best_centers = np.stack([cx, cy], axis=1).copy()
    best_sides = sides.copy()

    # Variables: [cx0, cy0, s0, cx1, cy1, s1, ...]
    def flatten(centers, sides):
        return np.concatenate([centers[:, 0], centers[:, 1], sides])

    def unflatten(x):
        cx = x[0:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        return cx, cy, s

    def objective(x):
        cx, cy, s = unflatten(x)
        # Penalty for out of bounds
        # s must be > 0
        # cx - s/2 >= 0 => s <= 2*cx
        # 1 - cx - s/2 >= 0 => s <= 2*(1-cx)
        # So s <= 2 * min(cx, 1-cx)
        
        # We use a soft penalty for bounds to allow the optimizer to explore
        # but we want to enforce them strictly in the final check.
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # x overlap
                dx = abs(cx[i] - cx[j])
                limit_x = (s[i] + s[j]) / 2.0
                if dx < limit_x:
                    ox = limit_x - dx
                else:
                    ox = 0.0
                # y overlap
                dy = abs(cy[i] - cy[j])
                limit_y = (s[i] + s[j]) / 2.0
                if dy < limit_y:
                    oy = limit_y - dy
                else:
                    oy = 0.0
                penalty += ox * oy
        
        # Bound penalty
        bound_pen = 0.0
        for i in range(n):
            if s[i] <= 0:
                bound_pen += (0.01 - s[i])**2
            # Left
            if cx[i] - s[i]/2 < 0:
                bound_pen += (cx[i] - s[i]/2)**2
            # Right
            if cx[i] + s[i]/2 > 1:
                bound_pen += (cx[i] + s[i]/2 - 1)**2
            # Bottom
            if cy[i] - s[i]/2 < 0:
                bound_pen += (cy[i] - s[i]/2)**2
            # Top
            if cy[i] + s[i]/2 > 1:
                bound_pen += (cy[i] + s[i]/2 - 1)**2
        
        # We want to maximize sum(s), so minimize -sum(s)
        return -np.sum(s) + 100.0 * (penalty + bound_pen)

    # Bounds for optimizer
    # cx, cy in [0, 1]
    # s in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
    for i in range(n):
        bounds.append((0.0, 1.0)) # cy
    for i in range(n):
        bounds.append((0.01, 1.0)) # s (keep > 0)

    # Run local optimization from a few starting points
    starting_points = []
    
    # 1. The grid
    starting_points.append(flatten(best_centers, best_sides))
    
    # 2. Jittered grids
    rng = np.random.default_rng(42)
    for _ in range(5):
        cx_j = cx + rng.uniform(-0.05, 0.05, n)
        cy_j = cy + rng.uniform(-0.05, 0.05, n)
        s_j = sides + rng.uniform(-0.05, 0.05, n)
        # Clip to valid range roughly
        cx_j = np.clip(cx_j, 0, 1)
        cy_j = np.clip(cy_j, 0, 1)
        s_j = np.clip(s_j, 0.1, 1.0)
        starting_points.append(flatten(np.stack([cx_j, cy_j], axis=1), s_j))

    # 3. A known better configuration guess?
    # Let's try a 2x5 grid of 0.2 squares? No, sum is 2.0.
    # Let's try random uniform
    for _ in range(5):
        cx_r = rng.uniform(0.1, 0.9, n)
        cy_r = rng.uniform(0.1, 0.9, n)
        s_r = rng.uniform(0.2, 0.4, n)
        starting_points.append(flatten(np.stack([cx_r, cy_r], axis=1), s_r))

    for x0 in starting_points:
        if time.time() - start_time > time_limit:
            break
            
        # Use Powell for robustness with non-smooth penalties
        res = minimize(objective, x0, method='Powell', bounds=bounds, 
                       options={'maxiter': 5000, 'disp': False})
        
        if res.success or res.fun < 0: # res.fun is -sum + penalty. If penalty is small, fun is negative.
            cx, cy, s = unflatten(res.x)
            
            # Check validity
            valid = True
            if np.any(s <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(cx)) or np.any(~np.isfinite(cy)):
                valid = False
            
            if valid:
                # Check bounds
                if np.any(cx - s/2 < -1e-7) or np.any(cx + s/2 > 1 + 1e-7) or \
                   np.any(cy - s/2 < -1e-7) or np.any(cy + s/2 > 1 + 1e-7):
                    valid = False
            
            if valid:
                # Check overlaps
                for i in range(n):
                    for j in range(i+1, n):
                        t = (s[i] + s[j]) / 2.0 - 1e-7
                        if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_centers = np.stack([cx, cy], axis=1)
                    best_sides = s

    # Final check and return
    if best_score < 10/3.0:
        # Fallback to grid if optimization failed
        return best_centers, best_sides
    else:
        return best_centers, best_sides

# EVOLVE-BLOCK-END
