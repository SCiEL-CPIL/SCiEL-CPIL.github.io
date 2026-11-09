# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smooth_overlap_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth penalty for overlaps.
    Uses a soft-max approximation for the max(0, overlap) terms.
    """
    n = len(sides)
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t <= 0:
                continue
            ox = t - dx
            oy = t - dy
            # We want to penalize if ox > 0 AND oy > 0.
            # This is equivalent to max(0, ox) * max(0, oy)
            # Smooth max(0, x) approx: (x + sqrt(x^2 + eps)) / 2 is not smooth at 0 in derivative? 
            # log-sum-exp style: soft_max(x, 0) = (1/beta) * log(exp(beta*x) + 1)
            # Let's use a simple quadratic hinge which is C1 continuous:
            # max(0, x)^2 is C1.
            # But we need the product to be non-zero only when both are positive.
            # If we just sum max(0, ox)^2 + max(0, oy)^2, it penalizes separation in one axis even if the other is fine.
            # We need a penalty that is 0 if dx >= t OR dy >= t.
            # Let A = max(0, t - dx), B = max(0, t - dy).
            # Penalty = A * B is not smooth at A=0 or B=0.
            # Smoothed: soft_A = (A + sqrt(A^2 + 1e-12)) / 2? No, that's always positive.
            # We want soft_max(0, A).
            # Let f(x) = (x + sqrt(x^2 + eps)) / 2. This is smooth and approximates max(0, x).
            # f(0) = sqrt(eps)/2.
            # To make it 0 at 0, we can subtract f(0)? Or use a different smooth max.
            # g(x) = (x + sqrt(x^2 + eps)) / 2 - sqrt(eps)/2.
            # g(0) = 0. g'(0) = 0.5.
            # This is C1 smooth.
            eps = 1e-12
            soft_A = (ox + np.sqrt(ox*ox + eps)) / 2.0 - np.sqrt(eps)/2.0
            soft_B = (oy + np.sqrt(oy*oy + eps)) / 2.0 - np.sqrt(eps)/2.0
            if soft_A > 0 and soft_B > 0:
                pen += soft_A * soft_B
    return pen

def _boundary_penalty(centers, sides, tol=1e-7):
    """Penalty for squares going outside [0,1]^2."""
    pen = 0.0
    for i in range(len(sides)):
        cx, cy = centers[i]
        s = sides[i]
        # Smooth max(0, violation)
        # Violation left: -tol - (cx - s/2)
        v_l = -tol - (cx - s/2)
        v_r = (cx + s/2) - (1 + tol)
        v_b = -tol - (cy - s/2)
        v_t = (cy + s/2) - (1 + tol)
        
        def smooth_max_0(x):
            eps = 1e-12
            return (x + np.sqrt(x*x + eps)) / 2.0 - np.sqrt(eps)/2.0
        
        pen += smooth_max_0(v_l)**2
        pen += smooth_max_0(v_r)**2
        pen += smooth_max_0(v_b)**2
        pen += smooth_max_0(v_t)**2
    return pen

def _objective(params, n=10):
    """Objective: minimize -sum(sides) + penalties."""
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Ensure sides are positive (clip for calculation, but penalty will push them up)
    # If sides are negative, the overlap logic breaks. 
    # We assume sides > 0 in the valid region.
    sides_clipped = np.maximum(sides, 1e-6)
    
    obj = -np.sum(sides_clipped)
    # Weight for overlap penalty needs to be high enough to prevent overlaps
    # but not so high that it dominates the gradient and prevents side growth.
    obj += 50.0 * _smooth_overlap_penalty(centers, sides_clipped)
    obj += 50.0 * _boundary_penalty(centers, sides_clipped)
    return obj

def _is_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # --- Initial Configurations ---
    
    # Config 1: 2 large (0.5), 8 small (0.25)
    # Layout:
    # [ 0.5 ] [ 0.5 ]
    # [0.25 0.25][0.25 0.25]  <- Top half? No.
    # Let's define:
    # Top Left: 0.5x0.5 at (0.25, 0.75)
    # Top Right: 0.5x0.5 at (0.75, 0.75)
    # Bottom Left: 4 squares of 0.25 in 0.5x0.5 area
    # Bottom Right: 4 squares of 0.25 in 0.5x0.5 area
    
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    
    # Bottom Left Quadrant (0 to 0.5, 0 to 0.5)
    # 4 squares of 0.25
    c1[2] = [0.125, 0.375]; s1[2] = 0.25
    c1[3] = [0.375, 0.375]; s1[3] = 0.25
    c1[4] = [0.125, 0.125]; s1[4] = 0.25
    c1[5] = [0.375, 0.125]; s1[5] = 0.25
    
    # Bottom Right Quadrant (0.5 to 1.0, 0 to 0.5)
    c1[6] = [0.625, 0.375]; s1[6] = 0.25
    c1[7] = [0.875, 0.375]; s1[7] = 0.25
    c1[8] = [0.625, 0.125]; s1[8] = 0.25
    c1[9] = [0.875, 0.125]; s1[9] = 0.25
    
    update_best(c1, s1)
    
    # Config 2: 3x3 grid (9 squares of 1/3) + 1 extra? 
    # Actually, just start with 9 of 1/3 and one random small one, let optimizer fix it.
    # Or 10 squares in a 2x5 grid?
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c2[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s2[i] = 0.2
    # This is 2 rows of 5 squares of side 0.2. Sum = 2.0. Bad start.
    # Better: 2x5 grid of squares with side 0.5/2 = 0.25? No, 5 * 0.25 = 1.25 > 1.
    # 2x5 grid of squares side 0.2 is valid.
    
    # Config 3: 3x3 grid of 1/3, plus one small square squeezed in?
    # Start with 9 squares of 1/3, and 1 square of 0.1 in the center?
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    idx = 0
    for r in range(3):
        for c in range(3):
            c3[idx] = [(c + 0.5)/3.0, (r + 0.5)/3.0]
            s3[idx] = 1.0/3.0
            idx += 1
    # 10th square: place it in the center, overlapping? No, must be valid.
    # The 3x3 grid fills the square. No room for a 10th square of size > 0.
    # So this is an invalid start. We need to shrink the 9 squares to make room.
    # Let's scale the 3x3 grid by 0.9 to make room in the center.
    for i in range(9):
        c3[i] = c3[i] * 0.9 + 0.5 * 0.1 # Shift to center of 0.9 box?
        # Center of 0.9 box is (0.5, 0.5).
        # Original centers were (1/6, 1/6) etc.
        # Let's just create a 3x3 grid of side 0.3 (sum 2.7) and one square of 0.1?
        # 3*0.3 = 0.9 width.
        c3 = np.zeros((n, 2))
        s3 = np.zeros(n)
        idx = 0
        for r in range(3):
            for c in range(3):
                c3[idx] = [0.05 + (c + 0.5)*0.3, 0.05 + (r + 0.5)*0.3]
                s3[idx] = 0.3
                idx += 1
        # Center hole is 0.1x0.1?
        # Gap between 0.3 squares: 0.05 margin.
        # Center square of 0.1 fits in the 0.1 gap?
        # Center of grid is (0.5, 0.5).
        c3[9] = [0.5, 0.5]
        s3[9] = 0.1
        update_best(c3, s3)

    # Random Restarts
    np.random.seed(42)
    
    # Define a set of "biased" random starts
    # 1. Perturb Config 1
    # 2. Perturb Config 3
    # 3. Pure Random with size constraints
    
    starts = []
    starts.append((c1.copy(), s1.copy()))
    starts.append((c3.copy(), s3.copy()))
    
    # Add random perturbations of Config 1
    for _ in range(10):
        c_r = c1.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s1.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.1, 0.6)
        starts.append((c_r, s_r))
        
    # Add random perturbations of Config 3
    for _ in range(10):
        c_r = c3.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s3.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.05, 0.5)
        starts.append((c_r, s_r))

    # Pure random starts
    for _ in range(20):
        c_r = np.random.uniform(0.1, 0.9, (n, 2))
        s_r = np.random.uniform(0.1, 0.4, n)
        starts.append((c_r, s_r))

    for c_init, s_init in starts:
        if time.time() > deadline:
            break
            
        # Check if start is valid, if not, we still try to optimize, 
        # but SLSQP might struggle if it starts deeply infeasible.
        # We rely on the smooth penalty to guide it back.
        
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            res = minimize(_objective, params_init, args=(n,), method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
            
            if res.success or res.fun < _objective(params_init, n):
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                
                update_best(c_opt, s_opt)
                
                # Polish step: If we found a valid solution, try to polish it with tighter tolerance
                if _is_valid(c_opt, s_opt):
                    res_polish = minimize(_objective, params_opt, args=(n,), method='SLSQP',
                                          options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
                    if res_polish.success:
                        params_pol = res_polish.x
                        c_pol = params_pol[:2*n].reshape(n, 2)
                        s_pol = np.maximum(params_pol[2*n:], 1e-6)
                        update_best(c_pol, s_pol)
                        
        except Exception:
            pass

    if best_centers is None:
        # Fallback to Config 1
        best_centers = c1
        best_sides = s1

    # Final Polish: If we have a valid solution, try to improve it further with a dedicated high-precision pass
    if best_centers is not None and time.time() < deadline - 10:
        params_best = np.concatenate([best_centers.flatten(), best_sides])
        
        try:
            # Use a very tight tolerance and more iterations for the final polish
            res_final = minimize(_objective, params_best, args=(n,), method='SLSQP',
                                 options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
            
            if res_final.success or res_final.fun < _objective(params_best, n):
                params_final = res_final.x
                c_final = params_final[:2*n].reshape(n, 2)
                s_final = np.maximum(params_final[2*n:], 1e-6)
                
                if _is_valid(c_final, s_final):
                    score_final = np.sum(s_final)
                    if score_final > best_score:
                        best_score = score_final
                        best_centers = c_final
                        best_sides = s_final
                        
        except Exception:
            pass

    return best_centers, best_sides

# EVOLVE-BLOCK-END
