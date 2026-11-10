# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Optimizes the placement of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses L-BFGS-B with a smooth penalty function for constraints, combined with a final high-precision polish.
    """
    n = 10
    start_time = time.time()
    max_time = 580.0  # Safety margin before 600s limit

    def get_bounds(x):
        # State representation: x[0:n] = cx, x[n:2n] = cy, x[2n:3n] = s
        cx = x[:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        return cx, cy, s

    def check_validity(cx, cy, s, tol=1e-7):
        if np.any(cx < -tol) or np.any(cx > 1 + tol):
            return False
        if np.any(cy < -tol) or np.any(cy > 1 + tol):
            return False
        if np.any(s <= 0):  # Side must be > 0
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0 - tol
                min_dy = (s[i] + s[j]) / 2.0 - tol
                if dx < min_dx and dy < min_dy:
                    return False
        return True

    def calculate_violation(cx, cy, s, tol=1e-7):
        penalty = 0.0
        # Boundary violations (squared for smoothness)
        v = np.maximum(0, s/2 - cx)
        penalty += np.sum(v**2)
        v = np.maximum(0, cx + s/2 - 1)
        penalty += np.sum(v**2)
        v = np.maximum(0, s/2 - cy)
        penalty += np.sum(v**2)
        v = np.maximum(0, cy + s/2 - 1)
        penalty += np.sum(v**2)
        
        # Overlap violations (smooth approximation)
        # Use a soft penalty that encourages separation
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                min_dx = (s[i] + s[j]) / 2.0
                min_dy = (s[i] + s[j]) / 2.0
                
                ox = max(0, min_dx - dx)
                oy = max(0, min_dy - dy)
                
                # Penalty is proportional to the area of overlap if both axes overlap
                # Using squared terms for smoother gradient
                if ox > 0 and oy > 0:
                    penalty += (ox * oy) ** 2
                else:
                    # If only one axis overlaps, still penalize to encourage separation
                    penalty += 0.1 * (ox**2 + oy**2)
        return penalty

    def objective(x):
        cx, cy, s = get_bounds(x)
        if np.any(s < 0):
            return 1e10
        violation = calculate_violation(cx, cy, s)
        score = np.sum(s)
        # We want to maximize score, minimize violation.
        # Large weight on violation to ensure validity
        return 1e6 * violation - score

    def generate_initial_config(seed_val):
        np.random.seed(seed_val)
        
        # Template 1: 2x5 grid
        if seed_val % 2 == 0:
            cols = [0.1, 0.3, 0.5, 0.7, 0.9]
            rows = [0.2, 0.8]
            cx = []
            cy = []
            s = []
            for r in rows:
                for c in cols:
                    cx.append(c)
                    cy.append(r)
                    s.append(0.2)
            x = np.zeros(3*n)
            x[:n] = cx
            x[n:2*n] = cy
            x[2*n:3*n] = s
            return x
            
        # Template 2: Random valid-ish packing
        cx = np.random.rand(n)
        cy = np.random.rand(n)
        s = np.random.rand(n) * 0.25 + 0.05
        
        # Quick repulsion pass to reduce initial overlaps
        for _ in range(50):
            for i in range(n):
                for j in range(i+1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 0.5:
                        # Push apart
                        if dist > 1e-9:
                            cx[i] += 0.01 * dx / dist
                            cy[i] += 0.01 * dy / dist
                            cx[j] -= 0.01 * dx / dist
                            cy[j] -= 0.01 * dy / dist
            cx = np.clip(cx, 0.05, 0.95)
            cy = np.clip(cy, 0.05, 0.95)
            
        x = np.zeros(3*n)
        x[:n] = cx
        x[n:2*n] = cy
        x[2*n:3*n] = s
        return x

    best_score = -np.inf
    best_x = None
    
    # Run multiple restarts with L-BFGS-B
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time - 30:
            break
            
        x0 = generate_initial_config(r)
        
        # Bounds for L-BFGS-B
        # cx, cy in [0, 1]
        # s in [0.01, 1]
        bounds = [(0.0, 1.0)] * (2*n) + [(0.01, 1.0)] * n
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            x_opt = res.x
        except Exception:
            continue
            
        cx, cy, s = get_bounds(x_opt)
        
        # Check validity
        if check_validity(cx, cy, s):
            val = np.sum(s)
            if val > best_score:
                best_score = val
                best_x = x_opt.copy()
                
    # Final Polish: Use a tighter tolerance and more iterations on the best solution
    if best_x is not None:
        # Refine with Nelder-Mead or a tighter L-BFGS-B
        bounds = [(0.0, 1.0)] * (2*n) + [(0.01, 1.0)] * n
        
        # Try a few more steps with higher precision
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10})
            x_opt = res.x
            cx, cy, s = get_bounds(x_opt)
            if check_validity(cx, cy, s):
                val = np.sum(s)
                if val > best_score:
                    best_score = val
                    best_x = x_opt.copy()
        except Exception:
            pass

        # Manual hill-climbing polish: try to increase sides slightly
        cx, cy, s = get_bounds(best_x)
        current_val = np.sum(s)
        
        for _ in range(5000):
            if time.time() - start_time > max_time:
                break
            idx = np.random.randint(0, 3*n)
            if idx >= 2*n:
                # Increase side
                delta = np.random.uniform(0.0001, 0.001)
            else:
                # Move center
                delta = np.random.normal(0, 0.001)
                
            x_new = best_x.copy()
            x_new[idx] += delta
            if idx >= 2*n:
                x_new[idx] = max(0.01, x_new[idx])
            else:
                x_new[idx] = np.clip(x_new[idx], 0, 1)
                
            cx_n, cy_n, s_n = get_bounds(x_new)
            if check_validity(cx_n, cy_n, s_n):
                new_val = np.sum(s_n)
                if new_val > current_val:
                    best_x = x_new
                    current_val = new_val
                    cx, cy, s = get_bounds(best_x)

    if best_x is None:
        # Fallback to seed
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        return centers, sides

    cx, cy, s = get_bounds(best_x)
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    # Final safety check
    if not check_validity(cx, cy, sides):
        k = 4
        side = 0.25
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        
    return centers, sides

# EVOLVE-BLOCK-END
