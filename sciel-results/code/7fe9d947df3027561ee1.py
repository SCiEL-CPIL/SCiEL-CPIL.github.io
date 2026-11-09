# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths.
    
    Strategy:
    1. Global Optimization with L-BFGS-B.
    2. Smooth objective function: -sum(sides) + penalty(overlaps) + penalty(boundaries).
       - Overlap penalty uses softplus to be differentiable.
       - Boundary penalty ensures squares stay in [0,1]^2.
    3. Multi-start initialization:
       - Start 1: "2-large/8-small" topology (2 big squares in corners, 8 small filling rest).
       - Start 2: Uniform grid (2x5).
       - Start 3-10: Random feasible perturbations.
    4. Return the best feasible solution found.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 610  # Leave 20s margin for safety
    
    # Variables: x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
    # Shape: (30,)
    
    def unpack(x):
        centers = x[0:2*n:3].reshape(n, 2)
        sides = x[2::3]
        return centers, sides

    def objective(x):
        centers, sides = unpack(x)
        
        # 1. Maximize sum of sides (minimize negative sum)
        obj = -np.sum(sides)
        
        # 2. Boundary penalties
        # Squares must be in [0, 1]
        # Left/Bottom: cx - s/2 >= 0  => cx >= s/2
        # Right/Top: cx + s/2 <= 1   => cx <= 1 - s/2
        # Penalty if violated
        # We use a soft penalty for the objective to guide the optimizer,
        # but we also use hard bounds in L-BFGS-B where possible.
        # However, bounds on x depend on s, which is also a variable.
        # So we must rely on penalty for boundaries.
        
        # Violation: min(0, cx - s/2) and min(0, 1 - cx - s/2)
        # Let's define v_left = s/2 - cx, v_right = cx + s/2 - 1
        # Penalty = sum(softplus(v_left)^2 + softplus(v_right)^2)
        
        v_left_x = sides/2 - centers[:, 0]
        v_right_x = centers[:, 0] + sides/2 - 1
        v_left_y = sides/2 - centers[:, 1]
        v_right_y = centers[:, 1] + sides/2 - 1
        
        # Softplus approximation: log(1 + exp(x))
        # For stability, use np.logaddexp(0, x)
        # We want to penalize positive violations.
        # If v > 0, penalty is large.
        # Use softplus(v) * v to make it quadratic-like for large v? 
        # Or just softplus(v). Let's use softplus(v) * 10 to weight it.
        
        pen_boundary = (np.sum(np.logaddexp(0, 10 * v_left_x)) + 
                        np.sum(np.logaddexp(0, 10 * v_right_x)) + 
                        np.sum(np.logaddexp(0, 10 * v_left_y)) + 
                        np.sum(np.logaddexp(0, 10 * v_right_y)))
        
        # 3. Overlap penalties
        # For each pair (i, j), if they overlap, penalty.
        # Overlap condition: |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
        # Let dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
        # Let req_x = (s_i + s_j)/2, req_y = (s_i + s_j)/2
        # Overlap amount in x: max(0, req_x - dx)
        # Overlap amount in y: max(0, req_y - dy)
        # If both > 0, they overlap.
        # Penalty = sum(softplus(overlap_x) * softplus(overlap_y)) ?
        # Actually, if they overlap, we want to push them apart.
        # A common differentiable penalty for intersection of two boxes:
        # Let ox = max(0, req_x - dx)
        # Let oy = max(0, req_y - dy)
        # Penalty = softplus(ox) * softplus(oy)
        # This is 0 if either ox or oy is 0 (no overlap in that dimension).
        
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s_i = sides[i]
                s_j = sides[j]
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                
                req = (s_i + s_j) / 2
                
                ox = np.maximum(0, req - dx)
                oy = np.maximum(0, req - dy)
                
                # Use softplus to make max(0, x) differentiable
                # softplus(ox) is ~ ox for large ox, ~ log(2) for ox=0
                # We want 0 penalty when no overlap.
                # If we use softplus(10*ox) * softplus(10*oy), it's smooth.
                # But it adds a constant bias.
                # Better: penalize the product of the "depths" of overlap.
                # If ox > 0 and oy > 0, penalty = ox * oy.
                # To make differentiable: softplus(10*ox) * softplus(10*oy) - log(2)^2?
                # Let's just use a large weight on the product of the positive parts, smoothed.
                
                # Simpler: penalty = softplus(10 * (ox + oy)) ? No, that penalizes separation.
                # We want to penalize intersection area roughly.
                
                # Let's use:
                # p_x = softplus(10 * (req - dx))
                # p_y = softplus(10 * (req - dy))
                # Penalty = p_x * p_y
                # This is > 0 even if no overlap (due to log(2) bias).
                # Subtract the baseline: log(2)^2
                baseline = np.log(2)**2
                pen = p_x_val * p_y_val - baseline if False else 0 # placeholder
                
                # Let's refine:
                # We want 0 if ox<=0 or oy<=0.
                # f(x) = softplus(10x) is ~ 0 for x<<0, ~ 10x for x>>0.
                # So softplus(10*ox) is ~0 when no overlap in x.
                # Product is ~0 when no overlap.
                
                p_x = np.logaddexp(0, 10 * ox)
                p_y = np.logaddexp(0, 10 * oy)
                pen_overlap += p_x * p_y
        
        # Weight for penalties
        # The objective is -sum(sides) which is ~ -2.5 to -3.
        # Penalties should be large enough to dominate when infeasible.
        # Let's use a high weight.
        weight = 100.0
        obj += weight * (pen_boundary + pen_overlap)
        
        return obj

    def is_feasible(centers, sides):
        """Check if the configuration is valid."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
                return False
            if s <= 0:
                return False
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    def pack(centers, sides):
        """Pack centers and sides into x vector."""
        x = np.zeros(3 * n)
        x[0:2*n:3] = centers[:, 0]
        x[1:2*n:3] = centers[:, 1]
        x[2::3] = sides
        return x

    # Initializations
    
    # 1. 2-large/8-small
    # Two large squares in bottom-left and bottom-right?
    # Or top-left and bottom-right?
    # Let's try:
    # Large 1: (0.25, 0.25), s=0.5
    # Large 2: (0.75, 0.25), s=0.5
    # This fills the bottom half.
    # Top half: 8 small squares.
    # Grid 4x2 in top half?
    # y from 0.5 to 1.0. Height 0.5.
    # 2 rows of height 0.25.
    # 4 cols of width 0.25.
    # s=0.25.
    
    c_init = np.zeros((n, 2))
    s_init = np.zeros(n)
    
    # Large squares
    c_init[0] = [0.25, 0.25]
    s_init[0] = 0.5
    c_init[1] = [0.75, 0.25]
    s_init[1] = 0.5
    
    # Small squares
    idx = 2
    for row in range(2): # 0, 1
        for col in range(4): # 0, 1, 2, 3
            cy = 0.5 + 0.125 + row * 0.25
            cx = 0.125 + col * 0.25
            c_init[idx] = [cx, cy]
            s_init[idx] = 0.25
            idx += 1
            
    x0_1 = pack(c_init, s_init)
    
    # 2. Uniform 2x5 grid
    c_init2 = np.zeros((n, 2))
    s_init2 = np.zeros(n)
    for i in range(n):
        row = i // 5
        col = i % 5
        c_init2[i] = [0.1 + col * 0.2, 0.1 + row * 0.8]
        s_init2[i] = 0.2
    x0_2 = pack(c_init2, s_init2)
    
    # 3. Random starts
    rng = np.random.default_rng(42)
    x0_list = [x0_1, x0_2]
    for _ in range(8):
        c_r = rng.uniform(0.1, 0.9, (n, 2))
        s_r = rng.uniform(0.1, 0.3, n)
        x0_list.append(pack(c_r, s_r))
        
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    for x0 in x0_list:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue
            
        x_opt = res.x
        centers, sides = unpack(x_opt)
        
        if is_feasible(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers
                best_sides = sides
    
    # If no feasible solution found, return a known feasible one
    if best_centers is None:
        # Fallback: 2x5 grid
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            row = i // 5
            col = i % 5
            centers[i] = [0.1 + col * 0.2, 0.1 + row * 0.8]
            sides[i] = 0.2
        return centers, sides
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
