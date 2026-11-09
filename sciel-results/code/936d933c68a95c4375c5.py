# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0  # wall-clock safety margin (evaluator kills at 630s)

def _unpack_solution(x):
    """
    Unpack the flat array from the optimizer into centers and sides.
    x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    """
    centers = x.reshape(_N, 3)[:, :2]
    sides = x.reshape(_N, 3)[:, 2]
    return centers, sides

def _pack_solution(centers, sides):
    """
    Pack centers and sides into a flat array for the optimizer.
    """
    x = np.zeros(_N * 3)
    for i in range(_N):
        x[3*i] = centers[i, 0]
        x[3*i+1] = centers[i, 1]
        x[3*i+2] = sides[i]
    return x

def _objective(x):
    """
    Objective function for differential_evolution (minimization).
    Returns negative sum of sides + penalty for violations.
    """
    centers, sides = _unpack_solution(x)
    
    # Basic sanity checks
    if np.any(sides <= 0) or np.any(~np.isfinite(sides)) or np.any(~np.isfinite(centers)):
        return 1e6

    # Maximize sum of sides -> Minimize -sum(sides)
    obj = -np.sum(sides)
    
    # Penalty for boundary violations
    # Square i is inside [0,1] if:
    # cx_i - s_i/2 >= 0  => cx_i >= s_i/2
    # cx_i + s_i/2 <= 1  => cx_i <= 1 - s_i/2
    # Same for y.
    
    # We can compute violations directly from x to avoid unpacking overhead if needed, 
    # but unpacking is cheap compared to pairwise checks.
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    s = sides
    
    # Boundary violations
    # Violation is the amount by which the square extends outside [0,1]
    vx_low = np.maximum(0, s/2 - cx)
    vx_high = np.maximum(0, cx + s/2 - 1)
    vy_low = np.maximum(0, s/2 - cy)
    vy_high = np.maximum(0, cy + s/2 - 1)
    
    boundary_violation = np.sum(vx_low**2 + vx_high**2 + vy_low**2 + vy_high**2)
    
    # Pairwise overlap violations
    # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
    # We want to penalize the "depth" of overlap.
    # A common smooth penalty is the product of the penetration depths in x and y.
    
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    
    # Required separation in x to avoid overlap (with tolerance)
    req_x = 0.5 * (s[:, None] + s[None, :]) - _TOL
    req_y = 0.5 * (s[:, None] + s[None, :]) - _TOL
    
    # Penetration depth in x: how much they overlap in x (0 if separated)
    pen_x = np.maximum(0, req_x - dx)
    pen_y = np.maximum(0, req_y - dy)
    
    # Overlap volume (2D area proxy) is pen_x * pen_y
    # We sum these for all pairs i < j
    overlap_violation = np.sum(pen_x * pen_y)
    # Note: This counts each pair twice (i,j and j,i). We can divide by 2 or just let the penalty scale handle it.
    # Dividing by 2 is cleaner.
    overlap_violation /= 2.0
    
    # Total penalty
    # We need the penalty to be dominant when violations are significant.
    # Since sides are ~0.3, sum is ~3. 
    # If violation is 0.1, penalty should be >> 3.
    # Let's use a penalty factor of 1000.
    penalty = 1000.0 * (boundary_violation + overlap_violation)
    
    return obj + penalty

def _refine_solution(centers, sides, time_budget):
    """
    Local refinement using Nelder-Mead or similar to polish the solution
    and ensure strict feasibility.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    # Convert to flat array
    x0 = _pack_solution(centers, sides)
    
    # Define bounds for local search to keep it in the feasible box
    # cx in [s/2, 1-s/2], cy in [s/2, 1-s/2], s in [0.01, 0.5] (approx)
    # However, bounds depend on s, which is a variable. 
    # For local search, we can just rely on the penalty function being very strict.
    # Or we can fix bounds loosely.
    
    # Let's use a strict penalty function for local search
    def strict_obj(x):
        c, s = _unpack_solution(x)
        if np.any(s <= 1e-6):
            return 1e9
        cx, cy, sv = c[:,0], c[:,1], s
        
        # Boundary
        vx = np.maximum(0, sv/2 - cx) + np.maximum(0, cx + sv/2 - 1)
        vy = np.maximum(0, sv/2 - cy) + np.maximum(0, cy + sv/2 - 1)
        b_pen = np.sum(vx**2 + vy**2)
        
        # Overlap
        dx = np.abs(cx[:, None] - cx[None, :])
        dy = np.abs(cy[:, None] - cy[None, :])
        req = 0.5 * (sv[:, None] + sv[None, :]) - _TOL
        pen_x = np.maximum(0, req - dx)
        pen_y = np.maximum(0, req - dy)
        o_pen = 0.5 * np.sum(pen_x * pen_y)
        
        pen = 1e6 * (b_pen + o_pen)
        return -np.sum(sv) + pen

    # Nelder-Mead is derivative-free and good for small dimensions
    # Maxiter should be limited by time
    max_iter = 500
    
    res = minimize(strict_obj, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if res.success or res.fun < _objective(x0):
        best_x = res.x
    else:
        best_x = x0
        
    c, s = _unpack_solution(best_x)
    
    # Final clamp to ensure no numerical drift makes it infeasible
    # Clamp sides to be positive
    s = np.maximum(s, 1e-5)
    
    # Clamp centers to keep squares inside [0,1]
    # cx in [s/2, 1-s/2]
    cx = np.clip(c[:, 0], s * 0.5, 1.0 - s * 0.5)
    cy = np.clip(c[:, 1], s * 0.5, 1.0 - s * 0.5)
    
    return np.stack([cx, cy], axis=1), s

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize the sum of
    side lengths, with pairwise disjoint interiors (touching allowed).

    Returns:
        centers: np.ndarray of shape (10, 2), (x, y) center of each square.
        sides:   np.ndarray of shape (10,), side length of each square.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Define bounds for Differential Evolution
    # x: 0 to 1, y: 0 to 1, s: 0.01 to 0.5 (max possible for one square is 1, but with 10 squares, max side is much smaller)
    # Actually, max side for any single square in a packing of 10 is <= 1. 
    # But realistically, with 10 squares, the largest square is probably around 0.4-0.5.
    # Let's use wide bounds: [0, 1] for centers, [0.01, 0.5] for sides.
    # Wait, if one square is 0.5, it takes up a huge chunk. 
    # Let's allow s up to 0.6 to be safe, but the penalty will discourage it if it causes overlap.
    
    bounds = []
    for _ in range(_N):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 0.6)) # s

    # 2. Run Differential Evolution
    # We allocate ~500s to DE.
    de_deadline = start_time + 500.0
    
    best_centers = None
    best_sides = None
    best_obj_val = -np.inf
    
    try:
        # Use workers=-1 for parallelization if possible
        # If not supported, it will fall back to serial
        # popsize=15 is default, maybe increase to 20 for better exploration
        # maxiter=1000, but we will check time in the callback
        
        def callback(xk, convergence):
            nonlocal best_centers, best_sides, best_obj_val
            # Check time
            if time.time() > de_deadline:
                return True # Stop DE
            
            # Evaluate current best
            c, s = _unpack_solution(xk)
            # Quick feasibility check to update best if valid
            # The objective function includes penalties, so we need to check if penalty is ~0
            # For simplicity, we just track the objective value from DE's perspective
            # But we want to store the best *feasible* or *near-feasible* solution.
            
            # Let's just let DE run and take the final result.
            # However, DE might converge to a local minimum with high penalty.
            # It's better to track the best solution with low penalty.
            
            # To keep it simple and robust, we will just run DE to completion or timeout
            # and then refine the result.
            return False

        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=1000, 
            popsize=20, 
            tol=1e-8, 
            seed=42, 
            callback=callback,
            workers=-1
        )
        
        if res.success:
            best_x = res.x
        else:
            # If it didn't succeed, res.x might still be the best found
            best_x = res.x
            
        best_centers, best_sides = _unpack_solution(best_x)
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to a simple grid if DE crashes
        # Grid of 10 squares: 4x3 grid? 4 cols, 3 rows = 12 slots, use 10.
        # Side = 0.25
        # This is a safe fallback.
        best_centers = np.zeros((_N, 2))
        best_sides = np.zeros(_N)
        k = 4
        side = 0.25
        for i in range(_N):
            col = i % k
            row = i // k
            best_centers[i, 0] = (col + 0.5) * side
            best_centers[i, 1] = (row + 0.5) * side
            best_sides[i] = side

    # 3. Refine the solution
    # Allocate remaining time for refinement
    current_time = time.time()
    refine_budget = max(10.0, deadline - current_time - 5.0) # Leave 5s for final checks
    
    if refine_budget > 10:
        try:
            refined_centers, refined_sides = _refine_solution(best_centers, best_sides, refine_budget)
            
            # Check if refinement improved the objective (sum of sides) while maintaining feasibility
            # We need to verify feasibility strictly
            def is_feasible(c, s):
                if np.any(s <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(c)):
                    return False
                cx, cy = c[:, 0], c[:, 1]
                # Boundary
                if np.any(cx - s/2 < -_TOL) or np.any(cx + s/2 > 1 + _TOL):
                    return False
                if np.any(cy - s/2 < -_TOL) or np.any(cy + s/2 > 1 + _TOL):
                    return False
                # Overlap
                for i in range(_N):
                    for j in range(i+1, _N):
                        t = (s[i] + s[j])/2 - _TOL
                        if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                            return False
                return True

            if is_feasible(refined_centers, refined_sides):
                if np.sum(refined_sides) > np.sum(best_sides):
                    best_centers = refined_centers
                    best_sides = refined_sides
            else:
                # If refinement broke feasibility, try to clamp it
                s = np.maximum(refined_sides, 1e-5)
                cx = np.clip(refined_centers[:, 0], s * 0.5, 1.0 - s * 0.5)
                cy = np.clip(refined_centers[:, 1], s * 0.5, 1.0 - s * 0.5)
                c_clamped = np.stack([cx, cy], axis=1)
                if is_feasible(c_clamped, s):
                    if np.sum(s) > np.sum(best_sides):
                        best_centers = c_clamped
                        best_sides = s
        except Exception as e:
            print(f"Refinement failed: {e}")

    # 4. Final Validation and Fallback
    def is_feasible(c, s):
        if np.any(s <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(c)):
            return False
        cx, cy = c[:, 0], c[:, 1]
        # Boundary
        if np.any(cx - s/2 < -_TOL) or np.any(cx + s/2 > 1 + _TOL):
            return False
        if np.any(cy - s/2 < -_TOL) or np.any(cy + s/2 > 1 + _TOL):
            return False
        # Overlap
        for i in range(_N):
            for j in range(i+1, _N):
                t = (s[i] + s[j])/2 - _TOL
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    if not is_feasible(best_centers, best_sides):
        # Fallback to a known feasible solution
        # 2 large (0.45) + 8 small (0.25)
        centers = np.empty((_N, 2), dtype=np.float64)
        sides = np.empty(_N, dtype=np.float64)
        
        centers[0] = (0.225, 0.775)
        centers[1] = (0.775, 0.775)
        sides[0] = 0.45
        sides[1] = 0.45
        
        small = np.array([
            (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
            (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
        ], dtype=np.float64)
        centers[2:] = small
        sides[2:] = 0.25
        
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
