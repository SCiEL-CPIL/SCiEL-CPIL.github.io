import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # Safety margin before 300s limit
        
        call_count = 0
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            return float(func(x))
        
        rng = np.random.default_rng(42)
        dim = self.dim
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        
        # Phase 1: Space-filling initialization
        init_n = min(25, self.budget // 4)
        X_init = rng.uniform(-5, 5, (init_n, dim))
        y_init = np.array([wrapped_func(x) for x in X_init])
        
        best_x = X_init[np.argmin(y_init)]
        best_f = np.min(y_init)
        
        # Track all evaluated points
        X_eval = X_init.copy()
        y_eval = y_init.copy()
        
        def update_best(x_new, f_new):
            nonlocal best_x, best_f, X_eval, y_eval
            if f_new < best_f:
                best_x = x_new.copy()
                best_f = f_new
            X_eval = np.vstack([X_eval, [x_new]])
            y_eval = np.append(y_eval, f_new)
        
        # Phase 2: Adaptive coordinate descent with quadratic line-search
        # From the top K promising points
        K_starts = min(8, len(y_eval))
        idx_sorted = np.argsort(y_eval)[:K_starts]
        start_points = X_eval[idx_sorted]
        
        def coordinate_descent_quadratic(x0, max_iter=30):
            """
            Adaptive coordinate descent with quadratic line-search per axis.
            For each axis, fit a quadratic to 3 points and find the minimum.
            """
            x = x0.copy()
            f = wrapped_func(x)
            
            # Initial step sizes per axis
            step_sizes = np.full(dim, 0.5)
            
            for iteration in range(max_iter):
                if time.time() > deadline - 15:
                    break
                if call_count >= self.budget:
                    break
                
                improved = False
                for i in range(dim):
                    if time.time() > deadline - 15:
                        break
                    if call_count >= self.budget:
                        break
                    
                    # Try to find minimum along axis i
                    # Use quadratic interpolation with 3 points
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    
                    # Clip to bounds
                    x_minus[i] = np.clip(x_minus[i], lb[i], ub[i])
                    x_plus[i] = np.clip(x_plus[i], lb[i], ub[i])
                    
                    f_minus = wrapped_func(x_minus)
                    f_plus = wrapped_func(x_plus)
                    
                    # Check if we're at a boundary
                    at_lb = x[i] <= lb[i] + 1e-10
                    at_ub = x[i] >= ub[i] - 1e-10
                    
                    if at_lb and at_ub:
                        continue
                    
                    if at_lb:
                        # Only search in positive direction
                        if f_plus < f:
                            x[i] = x_plus[i]
                            f = f_plus
                            improved = True
                            step_sizes[i] *= 1.5
                        else:
                            step_sizes[i] *= 0.5
                    elif at_ub:
                        # Only search in negative direction
                        if f_minus < f:
                            x[i] = x_minus[i]
                            f = f_minus
                            improved = True
                            step_sizes[i] *= 1.5
                        else:
                            step_sizes[i] *= 0.5
                    else:
                        # Quadratic interpolation
                        h = step_sizes[i]
                        # Points: x-h, x, x+h with values f_minus, f, f_plus
                        # Fit quadratic: a*t^2 + b*t + c where t is offset from x
                        # f_minus = a*h^2 - b*h + c
                        # f = c
                        # f_plus = a*h^2 + b*h + c
                        a = (f_minus - 2*f + f_plus) / (2 * h * h) if h > 1e-10 else 0
                        b = (f_plus - f_minus) / (2 * h) if h > 1e-10 else 0
                        
                        if abs(a) > 1e-15:
                            t_min = -b / (2 * a)
                        else:
                            t_min = 0
                        
                        # Clip t_min to reasonable range
                        t_min = np.clip(t_min, -2*h, 2*h)
                        
                        x_new_i = x[i] + t_min
                        x_new_i = np.clip(x_new_i, lb[i], ub[i])
                        
                        if abs(t_min) < 1e-8:
                            step_sizes[i] *= 0.5
                            continue
                        
                        x_trial = x.copy()
                        x_trial[i] = x_new_i
                        f_trial = wrapped_func(x_trial)
                        
                        if f_trial < f:
                            x[i] = x_new_i
                            f = f_trial
                            improved = True
                            # Adapt step size based on how far we moved
                            step_sizes[i] = max(0.1 * abs(t_min), 0.05)
                        else:
                            step_sizes[i] *= 0.5
                            
                            # If step is too small, try a larger jump
                            if step_sizes[i] < 0.01:
                                step_sizes[i] = 0.5
                                # Try a random perturbation
                                x_rand = x.copy()
                                x_rand[i] = rng.uniform(lb[i], ub[i])
                                f_rand = wrapped_func(x_rand)
                                if f_rand < f:
                                    x[i] = x_rand[i]
                                    f = f_rand
                                    improved = True
                                    step_sizes[i] = 0.5
                
                if not improved:
                    break
                    
            return x, f
        
        # Run coordinate descent from each start point
        for start in start_points:
            if time.time() > deadline - 30:
                break
            if call_count >= self.budget:
                break
            
            x_cd, f_cd = coordinate_descent_quadratic(start, max_iter=20)
            update_best(x_cd, f_cd)
        
        # Phase 3: Multi-start L-BFGS-B polish
        # Select top candidates for polishing
        n_polish = min(5, len(y_eval))
        idx_sorted = np.argsort(y_eval)[:n_polish]
        polish_starts = X_eval[idx_sorted]
        
        remaining_budget = self.budget - call_count
        if remaining_budget > 10:
            for start in polish_starts:
                if time.time() > deadline - 10:
                    break
                if call_count >= self.budget - 5:
                    break
                
                try:
                    res = minimize(
                        wrapped_func,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': max(10, (self.budget - call_count) // 5), 
                                 'maxfun': max(10, (self.budget - call_count) // 3),
                                 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        return best_x, best_f
