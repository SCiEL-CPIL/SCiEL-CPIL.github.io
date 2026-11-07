import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills after 330s. 
        # The task says "Each evaluation must finish within 300 s".
        # We will aim for 280s to be safe.
        deadline = start_time + 280

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                try:
                    pt = pt.reshape(dim)
                except:
                    return best_f
            
            try:
                val = func(pt)
                if not np.isfinite(val):
                    return best_f
            except Exception:
                return best_f
                
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Sphere-Specific Model Construction
        # BBOB f1 is Sphere: f(x) = f_opt + sum((x_i - x*_i)^2)
        # This is equivalent to f(x) = a * ||x||^2 + q^T x + c
        # We can determine a and q using 1 + 2*dim points.
        
        d = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)
            
        # Evaluate the 1 + 2*dim points
        values = np.array([evaluate(p) for p in points])
        
        # If we ran out of budget or time during this phase, return best found
        if calls_made < len(points):
            return best_x, best_f

        c_val = values[0]
        a_vals = []
        q_vec = np.zeros(dim)
        
        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            
            # f(d*e_i) = a*d^2 + q_i*d + c
            # f(-d*e_i) = a*d^2 - q_i*d + c
            # Sum: 2*a*d^2 + 2*c = val_pos + val_neg  => a = (val_pos + val_neg - 2c) / (2*d^2)
            # Diff: 2*q_i*d = val_pos - val_neg       => q_i = (val_pos - val_neg) / (2*d)
            
            q_i = (val_pos - val_neg) / (2 * d)
            a_i = (val_pos + val_neg - 2*c_val) / (2 * d * d)
            
            a_vals.append(a_i)
            q_vec[i] = q_i
            
        # Average a to reduce numerical noise
        a = np.mean(a_vals)
        
        # If a is close to 0, the function is linear or flat. 
        # For Sphere, a should be > 0.
        if abs(a) < 1e-12:
            # Fallback: use a small positive value to avoid division by zero
            # This implies the function is very flat or we are in a linear region
            a = 1e-12 * (1 if np.mean(a_vals) >= 0 else -1)
            if a < 0:
                # If a is negative, it's a maximum, not a minimum. 
                # But Sphere is a minimum. If we got negative, something is wrong.
                # We will still try to minimize, but the "minimum" might be at the boundary.
                # However, BBOB f1 is convex. We assume a > 0.
                pass

        # Minimum of a*||x||^2 + q^T x + c is at x = -q / (2a)
        x_opt = -q_vec / (2 * a)
        
        # Clip to box [-5, 5]
        x_opt = np.clip(x_opt, -5, 5)
        
        # Evaluate the estimated optimum
        f_opt_est = evaluate(x_opt)
        
        # If we still have budget, polish with (1+1)-ES
        # The landscape is quadratic and isotropic (in the transformed space).
        # The "curvature" is 2a. 
        # A good initial step size is related to 1/sqrt(a).
        
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Initial sigma: scale by inverse sqrt of curvature
            # If a is large, the function is steep, we want smaller steps.
            # If a is small, the function is flat, we want larger steps.
            sigma_scale = 1.0 / np.sqrt(abs(a)) if abs(a) > 1e-6 else 1.0
            
            # Start with a modest step size
            sigma = 0.5 * sigma_scale
            
            rng = np.random.default_rng(12345)
            
            # (1+1)-ES Loop
            while calls_made < budget and time.time() < deadline:
                # Generate random step
                z = rng.standard_normal(dim)
                step = sigma * z
                
                y = x + step
                y = np.clip(y, -5, 5)
                
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Success: increase sigma slightly
                    sigma *= 1.1
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9
                    
                # Convergence check
                if sigma < 1e-12:
                    break

        return best_x, best_f
