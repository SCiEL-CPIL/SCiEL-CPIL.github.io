# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    Uses diverse initializations and L-BFGS-B optimization with smooth penalties.
    """
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin

    n = 10
    tol = 1e-7

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def objective(x):
        """
        x: [cx0, cy0, s0, cx1, cy1, s1, ...]
        Returns negative sum of sides + penalty.
        """
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Boundary penalties (soft)
        # We want s_i/2 <= cx_i <= 1 - s_i/2
        # Violation if cx_i - s_i/2 < 0 or cx_i + s_i/2 > 1
        pen = 0.0
        # Vectorized boundary penalty
        left_viol = np.maximum(0, -(cx - s/2))
        right_viol = np.maximum(0, (cx + s/2) - 1)
        bottom_viol = np.maximum(0, -(cy - s/2))
        top_viol = np.maximum(0, (cy + s/2) - 1)
        
        pen += np.sum(left_viol**2) * 1e6
        pen += np.sum(right_viol**2) * 1e6
        pen += np.sum(bottom_viol**2) * 1e6
        pen += np.sum(top_viol**2) * 1e6
        
        # Overlap penalties
        # For each pair i, j
        # t = (s_i + s_j)/2
        # overlap if |dx| < t and |dy| < t
        # Penalty: (t - |dx|)^2 if |dx| < t else 0, similarly for y
        # To make it smooth, we can use:
        # P = max(0, t - |dx|)^2 + max(0, t - |dy|)^2
        # This is C1 continuous but not C2 at 0. L-BFGS-B usually handles this.
        
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                
                # Smooth approximation of max(0, val)
                # val = t - abs(dx)
                # If val > 0, penalty is val^2
                # We can just use the max(0, ...) squared, it's differentiable except at 0.
                # To be safer, we can add a tiny epsilon or just rely on the optimizer.
                
                dx_pen = max(0, t - abs(dx))
                dy_pen = max(0, t - abs(dy))
                
                # If both are > 0, they overlap.
                # Penalty should be strong.
                pen += (dx_pen**2 + dy_pen**2) * 1e4
                
        return -np.sum(s) + pen

    def get_initial_guesses():
        guesses = []
        
        # 1. 3x3 grid + 1 small
        # 9 squares of 1/3, 1 square of 0.1
        cx = np.array([1/6, 1/2, 5/6, 1/6, 1/2, 5/6, 1/6, 1/2, 5/6, 0.5])
        cy = np.array([1/6, 1/6, 1/6, 1/2, 1/2, 1/2, 5/6, 5/6, 5/6, 0.5])
        s = np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 0.1])
        guesses.append(np.concatenate([cx, cy, s]))
        
        # 2. 2 large in corners, 4 medium, 4 small
        cx = np.array([0.25, 0.75, 0.25, 0.75, 0.5, 0.5, 0.25, 0.75, 0.5, 0.5])
        cy = np.array([0.75, 0.75, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.25, 0.25])
        s = np.array([0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        guesses.append(np.concatenate([cx, cy, s]))
        
        # 3. Uniform grid 4x4 (take 10)
        cx = np.array([0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375])
        cy = np.array([0.125, 0.125, 0.125, 0.125, 0.375, 0.375, 0.375, 0.375, 0.625, 0.625])
        s = np.full(10, 0.25)
        guesses.append(np.concatenate([cx, cy, s]))
        
        # 4. Random
        cx = np.random.uniform(0.1, 0.9, n)
        cy = np.random.uniform(0.1, 0.9, n)
        s = np.random.uniform(0.1, 0.3, n)
        guesses.append(np.concatenate([cx, cy, s]))
        
        # 5. "Spiral" like
        cx = np.array([0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.5, 0.5, 0.5, 0.5])
        cy = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5])
        s = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2])
        guesses.append(np.concatenate([cx, cy, s]))
        
        return guesses

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 0.5))  # s

    best_x = None
    best_val = 0.0
    
    initial_guesses = get_initial_guesses()
    
    # Add some random guesses
    for _ in range(5):
        cx = np.random.uniform(0.1, 0.9, n)
        cy = np.random.uniform(0.1, 0.9, n)
        s = np.random.uniform(0.1, 0.4, n)
        initial_guesses.append(np.concatenate([cx, cy, s]))

    for x_start in initial_guesses:
        if time.time() > deadline - 30:
            break
            
        # Clip to bounds
        x_start = x_start.copy()
        for i in range(n):
            x_start[3*i] = np.clip(x_start[3*i], 0.01, 0.99)
            x_start[3*i+1] = np.clip(x_start[3*i+1], 0.01, 0.99)
            x_start[3*i+2] = np.clip(x_start[3*i+2], 0.02, 0.49)
        
        try:
            result = minimize(
                objective, x_start, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            cx = result.x[0::3]
            cy = result.x[1::3]
            s = result.x[2::3]
            centers = np.column_stack([cx, cy])
            
            if is_valid(centers, s):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_x = result.x.copy()
                    
        except Exception:
            pass

    # If we have a best solution, do a few more refinement steps
    if best_x is not None:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            # Perturb best solution
            x_start = best_x + np.random.normal(0, 0.01, best_x.size)
            for i in range(n):
                x_start[3*i] = np.clip(x_start[3*i], 0.01, 0.99)
                x_start[3*i+1] = np.clip(x_start[3*i+1], 0.01, 0.99)
                x_start[3*i+2] = np.clip(x_start[3*i+2], 0.02, 0.49)
                
            try:
                result = minimize(
                    objective, x_start, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                cx = result.x[0::3]
                cy = result.x[1::3]
                s = result.x[2::3]
                centers = np.column_stack([cx, cy])
                
                if is_valid(centers, s):
                    val = np.sum(s)
                    if val > best_val:
                        best_val = val
                        best_x = result.x.copy()
            except Exception:
                pass

    if best_x is not None:
        cx = best_x[0::3]
        cy = best_x[1::3]
        s = best_x[2::3]
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = s.astype(np.float64)
        
        if is_valid(centers, sides):
            return centers, sides

    # Fallback: 4x4 grid (10 squares of side 0.25)
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
