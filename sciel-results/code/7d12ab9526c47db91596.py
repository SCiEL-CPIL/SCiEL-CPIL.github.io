"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    
    Approach:
    1. Use a smooth penalty function for overlaps and boundary violations.
    2. Optimize positions and radii using L-BFGS-B.
    3. The objective function balances maximizing the sum of radii against
       the penalty for constraint violations.
    """
    random.seed(42)
    
    def objective(params):
        """
        params: [w, x1, y1, r1, ..., x21, y21, r21]
        w: width of rectangle. Height h = 2 - w.
        """
        w = params[0]
        h = 2.0 - w
        
        # Ensure valid rectangle dimensions
        if w <= 0.1 or h <= 0.1:
            return 1e10
            
        circles = params[1:].reshape((-1, 3))
        # Ensure radii are positive
        radii = np.maximum(circles[:, 2], 1e-6)
        circles[:, 2] = radii
        
        penalty = 0.0
        
        # Boundary penalties (quadratic)
        for i in range(N):
            cx, cy, cr = circles[i]
            # Left
            if cx - cr < 0:
                penalty += (cr - cx) ** 2
            # Right
            if cx + cr > w:
                penalty += (cx + cr - w) ** 2
            # Bottom
            if cy - cr < 0:
                penalty += (cr - cy) ** 2
            # Top
            if cy + cr > h:
                penalty += (cy + cr - h) ** 2
        
        # Overlap penalties (quadratic)
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap ** 2
        
        # Objective: Maximize sum of radii, minimize penalty
        # We use a weighted sum. The weight for radii should be high enough
        # to encourage growth, but the penalty must dominate if constraints are violated.
        sum_radii = np.sum(radii)
        
        # Tuning parameter: How much do we care about radius vs penalty?
        # If penalty is 0, we want to maximize sum_radii.
        # If penalty > 0, we want to reduce it.
        # A common trick is to return penalty - weight * sum_radii.
        # However, we need to be careful with the scale.
        # Let's try a fixed weight.
        weight = 10.0
        
        return penalty - weight * sum_radii

    best_params = None
    best_score = -np.inf  # We are minimizing, so lower is better. But we track "score" as -objective.
    
    # Multiple restarts
    for restart in range(10):
        # Initialize w between 0.5 and 1.5 (symmetric around 1.0)
        w_init = 1.0 + (restart % 5 - 2) * 0.1
        h_init = 2.0 - w_init
        
        # Initialize circles in a grid
        side = 5 # 5x5 grid covers 21 easily
        init = [w_init]
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side * w_init
            y = (row + 0.5) / side * h_init
            r = min(w_init, h_init) / (2 * side) * 0.8
            init.extend([x, y, r])
            
        x0 = np.array(init)
        
        # Bounds:
        # w in [0.2, 1.8]
        # x in [0, w] -> but w changes. L-BFGS-B bounds are static.
        # This is a problem. We cannot have static bounds for x if w is a variable.
        # Solution: Use a penalty for out-of-bounds instead of hard bounds for x,y,r.
        # Or, fix w and optimize x,y,r? No, we want to optimize w too.
        # We will rely on the penalty function to keep things in bounds.
        # We must ensure the penalty is strong enough.
        
        # L-BFGS-B requires bounds. We can set loose bounds.
        # w: [0.2, 1.8]
        # x: [0, 2.0] (max possible width)
        # y: [0, 2.0] (max possible height)
        # r: [0.0, 1.0]
        
        bounds = [(0.2, 1.8)]  # w
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.0, 1.0)) # r
            
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        # Calculate the actual score (sum of radii) if valid
        # We need to check if the solution is valid (penalty ~ 0)
        w_opt = result.x[0]
        h_opt = 2.0 - w_opt
        circles = result.x[1:].reshape((-1, 3))
        
        # Check validity
        valid = True
        for i in range(N):
            cx, cy, cr = circles[i]
            if cx - cr < -1e-5 or cx + cr > w_opt + 1e-5 or cy - cr < -1e-5 or cy + cr > h_opt + 1e-5:
                valid = False
                break
            if cr <= 0:
                valid = False
                break
                
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((circles[i,0]-circles[j,0])**2 + (circles[i,1]-circles[j,1])**2)
                    if dist < circles[i,2] + circles[j,2] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_params = result.x
        else:
            # If invalid, we can try to fix it by shrinking radii slightly
            # But for now, we might skip this restart or try to repair.
            # Let's try a simple repair: scale down radii until valid.
            # This is a heuristic.
            scale = 1.0
            for _ in range(50):
                circles_test = circles.copy()
                circles_test[:, 2] *= scale
                ok = True
                for i in range(N):
                    cx, cy, cr = circles_test[i]
                    if cx - cr < -1e-5 or cx + cr > w_opt + 1e-5 or cy - cr < -1e-5 or cy + cr > h_opt + 1e-5:
                        ok = False
                        break
                if ok:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.sqrt((circles_test[i,0]-circles_test[j,0])**2 + (circles_test[i,1]-circles_test[j,1])**2)
                            if dist < circles_test[i,2] + circles_test[j,2] - 1e-5:
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    break
                scale *= 0.95
                
            if ok:
                score = np.sum(circles_test[:, 2])
                if score > best_score:
                    best_score = score
                    # Reconstruct best_params with scaled radii
                    new_params = np.array([w_opt] + circles_test.flatten())
                    best_params = new_params

    if best_params is None:
        # Fallback
        side = 5
        w = 1.0
        h = 1.0
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            circles[i, 2] = 0.5 / side * 0.8
        return circles

    w_opt = best_params[0]
    circles = best_params[1:].reshape((-1, 3))
    
    # Final cleanup to ensure strict validity
    # Clamp positions and radii
    h_opt = 2.0 - w_opt
    for i in range(N):
        cx, cy, cr = circles[i]
        cr = max(cr, 1e-6)
        cx = min(max(cx, cr), w_opt - cr)
        cy = min(max(cy, cr), h_opt - cr)
        circles[i] = [cx, cy, cr]
        
    # Resolve overlaps by shrinking
    for _ in range(20):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist - 1e-7:
                    # Shrink the smaller circle or both
                    excess = min_dist - dist
                    # Simple shrink
                    shrink = 0.5 * excess / (ri + rj)
                    circles[i, 2] = max(ri - shrink, 1e-6)
                    circles[j, 2] = max(rj - shrink, 1e-6)
                    valid = False
        if valid:
            break
            
    return circles
# EVOLVE-BLOCK-END
