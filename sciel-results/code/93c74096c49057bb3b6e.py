# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0
    
    def is_valid(params, strict_tol=1e-7):
        """
        params: [cx_0, cy_0, s_0, ..., cx_9, cy_9, s_9]
        Returns True if all constraints are satisfied.
        """
        if not np.all(np.isfinite(params)):
            return False
        
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        if np.any(s <= 0):
            return False
            
        # Boundary check
        for i in range(n):
            if cx[i] - s[i]/2 < -strict_tol:
                return False
            if cx[i] + s[i]/2 > 1 + strict_tol:
                return False
            if cy[i] - s[i]/2 < -strict_tol:
                return False
            if cy[i] + s[i]/2 > 1 + strict_tol:
                return False
                
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - strict_tol
                if t < 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    def penalized_objective(params, penalty_weight):
        """
        Minimize -sum(s) + penalty.
        """
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        obj = -np.sum(s)
        
        # Boundary penalties (push squares back inside)
        for i in range(n):
            # x bounds
            if cx[i] - s[i]/2 < 0:
                obj += penalty_weight * (0 - (cx[i] - s[i]/2))**2
            if cx[i] + s[i]/2 > 1:
                obj += penalty_weight * ((cx[i] + s[i]/2) - 1)**2
            # y bounds
            if cy[i] - s[i]/2 < 0:
                obj += penalty_weight * (0 - (cy[i] - s[i]/2))**2
            if cy[i] + s[i]/2 > 1:
                obj += penalty_weight * ((cy[i] + s[i]/2) - 1)**2
            # Side positivity
            if s[i] < 1e-8:
                obj += penalty_weight * (1e-8 - s[i])**2
                
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                # Overlap in x if dx < t, amount is t - dx
                ox = max(0, t - dx)
                # Overlap in y if dy < t, amount is t - dy
                oy = max(0, t - dy)
                
                # If both are overlapping, we have a conflict.
                # A smooth penalty that is zero if either ox or oy is 0.
                # Using product: if no overlap in x (ox=0) or no overlap in y (oy=0), penalty is 0.
                # If both overlap, penalty is proportional to area of intersection (ox*oy).
                # Squaring makes it smooth.
                obj += penalty_weight * (ox * oy)**2
                
        return obj

    def get_score(params):
        if not is_valid(params):
            return -np.inf
        s = params[2::3]
        return np.sum(s)

    # --- Initialization Strategies ---
    
    def init_guillotine(rng):
        """
        Generate a feasible packing using a guillotine cutting heuristic.
        This creates a diverse set of valid starting points.
        """
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Recursive function to fill a rectangle with k squares
        def fill_rect(x_min, y_min, x_max, y_max, k, idx_start):
            if k == 0:
                return
            w = x_max - x_min
            h = y_max - y_min
            
            if k == 1:
                # Place one square in the center of the rect, size limited by min(w,h)
                side = min(w, h)
                cx[idx_start] = (x_min + x_max) / 2
                cy[idx_start] = (y_min + y_max) / 2
                s[idx_start] = side
                return
            
            # Decide split direction: vertical or horizontal
            # Random choice, biased towards splitting the longer side or random
            if rng.random() < 0.5:
                # Vertical split
                # Choose split point
                # To ensure we can fit squares, split somewhat evenly
                split_ratio = rng.uniform(0.3, 0.7)
                x_split = x_min + w * split_ratio
                # Allocate squares
                k_left = int(np.round(k * split_ratio))
                k_left = max(1, min(k-1, k_left))
                k_right = k - k_left
                
                fill_rect(x_min, y_min, x_split, y_max, k_left, idx_start)
                fill_rect(x_split, y_min, x_max, y_max, k_right, idx_start + k_left)
            else:
                # Horizontal split
                split_ratio = rng.uniform(0.3, 0.7)
                y_split = y_min + h * split_ratio
                k_bottom = int(np.round(k * split_ratio))
                k_bottom = max(1, min(k-1, k_bottom))
                k_top = k - k_bottom
                
                fill_rect(x_min, y_min, x_max, y_split, k_bottom, idx_start)
                fill_rect(x_min, y_split, x_max, y_max, k_top, idx_start + k_bottom)
        
        fill_rect(0, 0, 1, 1, n, 0)
        
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        return params

    def init_grid(kx, ky):
        """
        Generate a grid packing.
        """
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Create a kx by ky grid
        # Total cells = kx * ky
        # We need n=10.
        # If kx*ky < 10, we can't use this simple grid.
        # If kx*ky >= 10, we fill the first 10 cells.
        
        if kx * ky < n:
            return None
            
        side = 1.0 / kx # Assume square cells? No, unit square.
        # If kx != ky, cells are rectangles. Squares must fit in them.
        # So max side is min(1/kx, 1/ky).
        side = min(1.0/kx, 1.0/ky)
        
        count = 0
        for r in range(ky):
            for c in range(kx):
                if count >= n:
                    break
                cx[count] = (c + 0.5) / kx
                cy[count] = (r + 0.5) / ky
                s[count] = side
                count += 1
            if count >= n:
                break
        
        # If we didn't fill 10 (shouldn't happen if kx*ky>=10), return None
        if count < n:
            return None
            
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        return params

    def init_random_feasible(rng):
        """
        Generate random small squares that don't overlap.
        """
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Start with very small squares and grow?
        # Or just place them randomly and check validity, retry if invalid.
        # Given time constraints, simple random placement of small squares is safer.
        max_side = 0.2
        for i in range(n):
            s[i] = rng.uniform(0.05, max_side)
            # Place in a valid spot? Hard to guarantee.
            # Just place randomly in [s/2, 1-s/2]
            cx[i] = rng.uniform(s[i]/2, 1 - s[i]/2)
            cy[i] = rng.uniform(s[i]/2, 1 - s[i]/2)
            
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        
        # This might be invalid. The optimizer will fix it.
        return params

    # --- Optimization Loop ---
    
    best_score = -np.inf
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # List of initializations to try
    inits = []
    
    # 1. Guillotine cuts (diverse and feasible)
    for _ in range(50):
        inits.append(init_guillotine(rng))
        
    # 2. Grids
    # 3x4 grid (12 cells, take 10)
    inits.append(init_grid(3, 4))
    # 4x3 grid
    inits.append(init_grid(4, 3))
    # 5x2 grid
    inits.append(init_grid(5, 2))
    # 2x5 grid
    inits.append(init_grid(2, 5))
    # 10x1 grid (very narrow, bad start but valid)
    inits.append(init_grid(10, 1))
    
    # 3. Random feasible-ish
    for _ in range(20):
        inits.append(init_random_feasible(rng))
        
    # Filter out None inits
    inits = [p for p in inits if p is not None]
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # cx
        bounds.append((0, 1))  # cy
        bounds.append((1e-8, 1.0))  # s
        
    # We will run L-BFGS-B on each init
    # To save time, we can skip inits that are clearly bad or take too long
    # L-BFGS-B is fast for this size.
    
    for i, init_params in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Check if init is already valid and good
        init_score = get_score(init_params)
        
        # Run optimizer
        # Use a moderate penalty weight. Too high -> numerical issues. Too low -> invalid result.
        # Start with 100.
        penalty_weight = 100.0
        
        try:
            res = minimize(
                lambda p: penalized_objective(p, penalty_weight),
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                score = get_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception:
            pass
            
        # If time allows, try a second pass with higher penalty on the best result so far
        # This helps "clean up" any residual overlaps
        if best_params is not None and time.time() < deadline - 30:
             if i % 10 == 0: # Do this periodically
                try:
                    res2 = minimize(
                        lambda p: penalized_objective(p, 1000.0),
                        best_params,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res2.success:
                        score2 = get_score(res2.x)
                        if score2 > best_score:
                            best_score = score2
                            best_params = res2.x.copy()
                except Exception:
                    pass

    # Fallback if no valid result found
    if best_params is None or not is_valid(best_params):
        # Use a simple 3x3 grid + 1 small square in the middle of one cell?
        # Or just a 4x4 grid with 10 squares.
        # 4x4 grid: side 0.25. Sum = 10 * 0.25 = 2.5.
        # This is a safe fallback.
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

    cx = best_params[0::3]
    cy = best_params[1::3]
    s = best_params[2::3]
    
    # Final check and cleanup
    # Ensure sides are strictly positive
    s = np.maximum(s, 1e-8)
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
