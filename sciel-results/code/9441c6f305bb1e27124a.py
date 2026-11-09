import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Pre-compute base corridor mask for the grid if we were at (0,0,0)?
    # No, the corridor moves. We need to check each point against the corridor.
    
    # To speed up evaluation, we can pre-compute the "active" points.
    # A point is active if it is inside the current intersection.
    
    def evaluate_path_fast(path, active_mask=None):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        active_mask: optional boolean mask of points to check (from previous step)
        Returns: (score, new_active_mask)
        """
        # Ensure path is a numpy array
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
        # The evaluator puts (0, 0, 0) before and (-2, 0, 90) after.
        # Our path contains intermediate points.
        # We need to simulate the evaluator's densification.
        
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        if active_mask is None:
            valid = np.ones(n_points, dtype=bool)
        else:
            valid = active_mask.copy()
            
        # If no points are valid, return 0
        if not np.any(valid):
            return 0.0, valid
            
        # We only check points that are currently valid
        # To be efficient, we can just check all points but mask the result?
        # No, checking 135k points * 90 angles is expensive.
        # Checking only valid points is better if the set shrinks.
        
        # However, vectorizing over a subset of P is tricky if we want to update the mask.
        # Let's just vectorize over all P but early exit if the count drops too low?
        # Actually, the current program checks all P for all angles.
        # Let's try to optimize the check itself.
        
        # For each placement, compute u, v for all P, then mask.
        # This is O(N * M) where N is points, M is placements.
        
        # Let's use the active mask to restrict computation if possible?
        # P[valid] is a copy. We can compute u, v for P[valid].
        # But then we need to map back to the full mask.
        
        # Let's stick to checking all points but use NumPy efficiently.
        # The bottleneck is the 90 iterations of trig and comparisons.
        
        # Optimization: Pre-compute cos/sin for all angles in dense_path?
        # dense_path is small (~90-100 points).
        
        # Let's just loop through dense_path and update valid.
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor:
            # Arm 1: 0 <= u <= 1, -3 <= v <= 1
            # Arm 2: -3 <= u <= 1, 0 <= v <= 1
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            # Early exit if too few points remain
            count = np.sum(valid)
            if count < 100: # Threshold to avoid wasting time on tiny areas
                return 0.0, valid
                
        score = np.sum(valid) * 0.0001
        return score, valid

    # Strategy:
    # Direct (x,y) parameterization at each integer degree from 1 to 89.
    # This gives 89 * 2 = 178 parameters.
    # Use coordinate descent with active set tracking.
    
    angles = np.arange(1, 90) # 1 to 89
    n_ctrl = len(angles)
    
    # Initial guess:
    # The sofa moves from (0,0) to (-2,0).
    # x(a) = -2 * a / 90
    # y(a) = h * sin(pi * a / 90)
    # Let's start with a reasonable guess.
    params = np.zeros(2 * n_ctrl)
    for i, a in enumerate(angles):
        t = a / 90.0
        params[2*i] = -2 * t
        params[2*i+1] = math.sin(math.pi * t) # Amplitude 1.0
        
    best_params = params.copy()
    best_score, best_mask = evaluate_path_fast(np.column_stack([params[0::2], params[1::2], angles]))
    
    # Coordinate Descent
    # We will iterate through parameters, adjusting one at a time.
    # To speed up, we can use a small step size and local search around current value.
    
    # Given the high dimensionality (178), pure coordinate descent might be slow.
    # But we have 600s.
    # Let's try a multi-scale approach.
    # Phase 1: Coarse search with larger steps on a subset of angles?
    # No, let's just do standard coordinate descent with a reasonable step size.
    
    # Actually, Nelder-Mead on 178 vars is likely to fail or be slow.
    # Coordinate descent is robust.
    
    # Let's define a helper to extract path from params
    def get_path(params):
        x = params[0::2]
        y = params[1::2]
        return np.column_stack([x, y, angles])
    
    # Optimization Loop
    # We will perform multiple passes.
    # In each pass, we try to improve each parameter.
    
    # Step size: Start with 0.1, reduce if no improvement.
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 5
    
    # To make it faster, we can parallelize the evaluation of neighbors?
    # No, just keep it simple.
    
    # We can also restrict the search space for x and y.
    # x should be between -3 and 1.
    # y should be between -1.5 and 1.5 (roughly).
    
    while time.time() < deadline - 10:
        improved = False
        for i in range(2 * n_ctrl):
            if time.time() > deadline - 10:
                break
                
            orig_val = params[i]
            best_local_val = orig_val
            best_local_score = best_score
            best_local_mask = best_mask
            
            # Try a few perturbations
            # Since the function is non-convex, we should try both + and -.
            # And maybe a larger step if we've been stuck.
            
            # Simple approach: try orig_val +/- step_size
            # If improvement, update and maybe keep going in that direction?
            # For simplicity, just try +/- step_size and keep the best.
            
            candidates = [orig_val - step_size, orig_val, orig_val + step_size]
            # Also try a larger jump if we are stuck?
            # Let's stick to local search for now.
            
            for c in candidates:
                # Clamp to reasonable bounds
                if i % 2 == 0: # x
                    c = np.clip(c, -3.5, 1.5)
                else: # y
                    c = np.clip(c, -2.0, 2.0)
                    
                if c == orig_val:
                    continue
                    
                params[i] = c
                path = get_path(params)
                score, mask = evaluate_path_fast(path, best_mask)
                
                if score > best_local_score + 1e-9:
                    best_local_score = score
                    best_local_val = c
                    best_local_mask = mask
                    improved = True
                    
            params[i] = best_local_val
            
            if best_local_score > best_score + 1e-9:
                best_score = best_local_score
                best_params = params.copy()
                best_mask = best_local_mask
            else:
                # If no improvement in this coordinate, revert
                pass
                
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Reduce step size
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-4:
                    break
        else:
            no_improve_count = 0
            # Optionally increase step size slightly if improving rapidly?
            # No, keep it stable.
            
    # Final evaluation with best_params
    final_path = get_path(best_params)
    final_score, _ = evaluate_path_fast(final_path)
    
    # Ensure output format is correct: float array of shape (n, 3)
    # The task says: returning a float array of shape (n, 3), n >= 1, all finite.
    # Row (x, y, a)
    
    return final_path
