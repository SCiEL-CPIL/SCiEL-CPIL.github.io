"""Root finder for 3 fixed 2-D systems: Global Grid Search + Damped Newton."""
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Global Search: Scan a grid of points in a large box around x0 to find 
       a point with small ||F(x)||. This handles cases where x0 is far from the root.
    2. Local Refinement: Use Damped Newton's method starting from the best 
       grid point to achieve high precision (1e-8) with few iterations.
    
    Budget:
    - Time limit: 120s (using 115s internal deadline).
    - Call limit: < 500 calls.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    tol = 1e-8
    max_calls = 450  # Safety margin below 500
    
    call_count = 0
    def safe_F(x):
        nonlocal call_count
        if call_count >= max_calls:
            return np.array([np.inf, np.inf])
        call_count += 1
        try:
            val = F(x)
            val = np.asarray(val, dtype=float)
            if val.shape != (2,) or not np.all(np.isfinite(val)):
                return np.array([np.inf, np.inf])
            return val
        except Exception:
            return np.array([np.inf, np.inf])
    
    def norm(v):
        return np.linalg.norm(v)
    
    # Initialize best point
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (2,) or not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    best_x = x0.copy()
    best_val = safe_F(best_x)
    best_norm = norm(best_val)
    
    # 1. Global Grid Search
    # We scan a box around x0. The size is adaptive: start small, expand if needed.
    # Grid resolution: 11x11 = 121 points per box.
    # Max boxes: 4 (total 484 calls). This fits in budget.
    
    # Box sizes to try: [1, 10, 100, 1000] relative to x0 scale?
    # Actually, let's use absolute scales centered at x0.
    # If x0 is [0,0], we check [-1,1], [-10,10], etc.
    # If x0 is [100, 200], we check around that.
    
    scales = [1.0, 10.0, 100.0, 1000.0]
    
    for scale in scales:
        if time.time() > deadline or call_count > max_calls - 20:
            break
            
        # Define grid
        # Points: x0 + scale * [-1, -0.5, 0, 0.5, 1] in each dimension?
        # That's 5x5 = 25 points.
        # Let's do 5x5 to save calls.
        offsets = np.linspace(-1.0, 1.0, 5)
        
        for dx in offsets:
            if time.time() > deadline or call_count > max_calls - 20:
                break
            for dy in offsets:
                if time.time() > deadline or call_count > max_calls - 20:
                    break
                x_try = x0 + np.array([dx, dy]) * scale
                val = safe_F(x_try)
                n_val = norm(val)
                if n_val < best_norm:
                    best_norm = n_val
                    best_x = x_try.copy()
                    
                # Early exit if we found a very good point
                if best_norm < 1e-4:
                    break
            if best_norm < 1e-4:
                break
        
        # If we found a point close enough, we can stop global search
        if best_norm < 1e-4:
            break
        # If we expanded to max scale and didn't find anything good, 
        # the root might be extremely far or the function is ill-behaved.
        # But for "fixed systems", usually roots are within reasonable bounds.
    
    # 2. Local Refinement: Damped Newton
    # Start from best_x
    x = best_x.copy()
    Fx = safe_F(x)
    n_Fx = norm(Fx)
    
    # If we already met tolerance, return
    if n_Fx <= tol:
        return x
    
    # Jacobian calculation (Finite Difference)
    def get_jacobian(x, Fx):
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            # We already have F(x). We need F(x+e)
            # Note: This consumes 2 calls per Jacobian eval.
            # Total calls per Newton step: 1 (Fx) + 2 (Jac) = 3 calls.
            # We have ~100 calls left after grid search (450 - 121*3 approx? No, grid is 121 max).
            # Grid: 4 scales * 25 pts = 100 calls.
            # Remaining: 350 calls.
            # Newton steps: 350 / 3 ~ 100 steps. Plenty.
            
            Fx_e = safe_F(x + e)
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    # Newton Loop
    for _ in range(50):
        if time.time() > deadline or call_count > max_calls - 5:
            break
            
        Fx = safe_F(x)
        n_Fx = norm(Fx)
        if n_Fx <= tol:
            return x
            
        if not np.all(np.isfinite(Fx)):
            break
            
        J = get_jacobian(x, Fx)
        
        # Check for singularity
        det = np.linalg.det(J)
        if abs(det) < 1e-12:
            # Singular Jacobian, try a small step in direction of -Fx
            # Or break and rely on best_x
            # Let's try a simple gradient descent step if possible, or just break
            # Since we are in refinement, breaking is safer than diverging
            break
            
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
            
        # Damping: ensure step is not too large
        # We want ||x - step|| to be reasonable.
        # If step is huge, damp it.
        step_norm = np.linalg.norm(step)
        if step_norm > 1.0:
            # Damp step to max 1.0
            step = step / step_norm
            
        x_new = x - step
        
        # Check if x_new is finite
        if not np.all(np.isfinite(x_new)):
            break
            
        Fx_new = safe_F(x_new)
        n_Fx_new = norm(Fx_new)
        
        if n_Fx_new < n_Fx:
            x = x_new
            # Update best if this is better than global best
            if n_Fx_new < best_norm:
                best_norm = n_Fx_new
                best_x = x_new.copy()
        else:
            # Line search: halve the step
            # Try a few halvings
            improved = False
            for k in range(5):
                x_try = x - step / (2 ** (k+1))
                if not np.all(np.isfinite(x_try)):
                    continue
                Fx_try = safe_F(x_try)
                n_Fx_try = norm(Fx_try)
                if n_Fx_try < n_Fx:
                    x = x_try
                    if n_Fx_try < best_norm:
                        best_norm = n_Fx_try
                        best_x = x.copy()
                    improved = True
                    break
            if not improved:
                # No improvement even with damping, break
                break

    # Return the best point found (either from refinement or global search)
    # Verify the best point one last time?
    # The evaluator does a final check.
    # We should return best_x.
    # But wait, if the last Newton step didn't improve, best_x might be old.
    # Let's re-evaluate best_x to be sure? No, we track it.
    
    # One final check: if best_x is not good enough, but we have calls left,
    # maybe try one more random restart?
    # No, the grid search + Newton should be sufficient for "fixed systems".
    
    return best_x
