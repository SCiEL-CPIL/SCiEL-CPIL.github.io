# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def squares_in_square10() -> tuple:
    """
    Pack 10 squares in a unit square to maximize the sum of their side lengths.
    Uses a penalty-based gradient ascent optimization with multiple random restarts.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    time_limit = 550.0  # Safety margin
    
    best_score = -1.0
    best_centers = None
    best_sides = None
    
    def compute_penalty_and_gradient(centers, sides):
        """
        Computes the penalty for constraint violations and the gradient of the penalty.
        Returns: penalty_value, grad_centers, grad_sides
        """
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Initialize gradients
        g_cx = np.zeros(n)
        g_cy = np.zeros(n)
        g_s = np.zeros(n)
        
        penalty = 0.0
        
        # Pairwise non-overlap
        # Soft-max approximation for max(u, v)
        beta = 50.0  # Sharpness of soft-max
        
        for i in range(n):
            for j in range(i + 1, n):
                r = (s[i] + s[j]) / 2.0
                
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                
                # u = r - |dx|
                # v = r - |dy|
                u = r - abs(dx)
                v = r - abs(dy)
                
                # Softmax(u, v)
                # To avoid overflow, use max
                m = max(u, v)
                if m > 500: 
                    sm = m
                    du = 1.0 if u == m else 0.0
                    dv = 1.0 if v == m else 0.0
                else:
                    exp_u = np.exp(beta * (u - m))
                    exp_v = np.exp(beta * (v - m))
                    denom = exp_u + exp_v
                    sm = (m + np.log(denom) / beta)
                    du = exp_u / denom
                    dv = exp_v / denom
                
                if sm > 0:
                    p = sm * sm
                    penalty += p
                    gp = 2 * sm
                    
                    # Gradients of sm w.r.t u and v
                    g_u = gp * du
                    g_v = gp * dv
                    
                    # Gradients of u and v w.r.t variables
                    # u = r - |dx|
                    # du/dcx_i = -sign(dx), du/dcx_j = sign(dx)
                    # du/ds_i = 0.5, du/ds_j = 0.5
                    sign_dx = 1.0 if dx > 0 else (-1.0 if dx < 0 else 0.0)
                    sign_dy = 1.0 if dy > 0 else (-1.0 if dy < 0 else 0.0)
                    
                    g_cx[i] += g_u * (-sign_dx)
                    g_cx[j] += g_u * (sign_dx)
                    g_cy[i] += g_v * (-sign_dy)
                    g_cy[j] += g_v * (sign_dy)
                    g_s[i] += 0.5 * (g_u + g_v)
                    g_s[j] += 0.5 * (g_u + g_v)
        
        # Boundary constraints
        # Inside [-tol, 1+tol]
        # x_i - s_i/2 >= -tol
        # x_i + s_i/2 <= 1+tol
        # y_i - s_i/2 >= -tol
        # y_i + s_i/2 <= 1+tol
        
        # Lower X
        viol_lx = -tol - (cx - s / 2.0)
        mask_lx = viol_lx > 0
        if np.any(mask_lx):
            p_lx = np.sum(viol_lx[mask_lx] ** 2)
            penalty += p_lx
            g_cx[mask_lx] -= 2 * viol_lx[mask_lx]
            g_s[mask_lx] += viol_lx[mask_lx]  # d(-tol - (cx - s/2))/ds = 0.5, wait.
            # viol = -tol - cx + s/2. d(viol)/ds = 0.5.
            # d(viol^2)/ds = 2*viol*0.5 = viol. Correct.
            
        # Upper X
        viol_ux = (cx + s / 2.0) - (1.0 + tol)
        mask_ux = viol_ux > 0
        if np.any(mask_ux):
            p_ux = np.sum(viol_ux[mask_ux] ** 2)
            penalty += p_ux
            g_cx[mask_ux] += 2 * viol_ux[mask_ux]
            g_s[mask_ux] += viol_ux[mask_ux]
            
        # Lower Y
        viol_ly = -tol - (cy - s / 2.0)
        mask_ly = viol_ly > 0
        if np.any(mask_ly):
            p_ly = np.sum(viol_ly[mask_ly] ** 2)
            penalty += p_ly
            g_cy[mask_ly] -= 2 * viol_ly[mask_ly]
            g_s[mask_ly] += viol_ly[mask_ly]
            
        # Upper Y
        viol_uy = (cy + s / 2.0) - (1.0 + tol)
        mask_uy = viol_uy > 0
        if np.any(mask_uy):
            p_uy = np.sum(viol_uy[mask_uy] ** 2)
            penalty += p_uy
            g_cy[mask_uy] += 2 * viol_uy[mask_uy]
            g_s[mask_uy] += viol_uy[mask_uy]
            
        return penalty, g_cx, g_cy, g_s

    def is_valid(centers, sides):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        if np.any(s <= 0):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        for i in range(n):
            for j in range(i + 1, n):
                r = (s[i] + s[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < r - tol and dy < r - tol:
                    return False
        return True

    def run_optimization(init_centers, init_sides):
        centers = init_centers.copy()
        sides = init_sides.copy()
        
        # Gradient Ascent
        # Objective: Max Sum(s) - Lambda * Penalty
        # Grad = Ones - Lambda * Grad_Penalty
        
        lambda_pen = 100.0
        step_size = 0.01
        max_iter = 1000
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            pen, g_cx, g_cy, g_s = compute_penalty_and_gradient(centers, sides)
            
            # Gradient of Objective
            # d/ds (Sum(s)) = 1
            # d/ds (Penalty) = g_s
            # Total Grad s = 1 - lambda * g_s
            # Total Grad cx = -lambda * g_cx
            # Total Grad cy = -lambda * g_cy
            
            grad_cx = -lambda_pen * g_cx
            grad_cy = -lambda_pen * g_cy
            grad_s = 1.0 - lambda_pen * g_s
            
            # Update
            centers[:, 0] += step_size * grad_cx
            centers[:, 1] += step_size * grad_cy
            sides += step_size * grad_s
            
            # Project to bounds (simple clipping)
            # s > 0
            sides = np.maximum(sides, 0.01)
            # Centers within [-1, 2] to allow penalty to work
            centers = np.clip(centers, -1.0, 2.0)
            
        # Final projection to ensure validity?
        # If invalid, try to shrink sides slightly
        if not is_valid(centers, sides):
             # Shrink all sides by 10%
             sides *= 0.9
             if not is_valid(centers, sides):
                 return -1.0, None, None
                 
        score = np.sum(sides)
        return score, centers, sides

    # Multiple random restarts
    n_restarts = 50
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        # Start with a rough grid or random
        if r == 0:
            # 3x3 grid + 1
            k = 3
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            # Fix the out-of-bounds one (idx 9)
            # Move it to center or shrink?
            # For init, just randomize slightly
            cx = cx + rng.uniform(-0.01, 0.01, n)
            cy = cy + rng.uniform(-0.01, 0.01, n)
            sides = np.full(n, side * 0.9)
        else:
            cx = rng.uniform(0.1, 0.9, n)
            cy = rng.uniform(0.1, 0.9, n)
            sides = rng.uniform(0.1, 0.3, n)
            
        centers = np.stack([cx, cy], axis=1)
        
        score, c, s = run_optimization(centers, sides)
        
        if score > best_score:
            best_score = score
            best_centers = c
            best_sides = s
            
    # If no valid solution found, return a valid fallback (e.g. 10 tiny squares)
    if best_centers is None:
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.01)
        return centers, sides
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
