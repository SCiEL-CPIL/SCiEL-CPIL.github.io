import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Structured Initialization
        # We want to cover the space [-4, 4]^5 reasonably well.
        # 1. Center
        init_points = [np.zeros(dim)]
        
        # 2. Axis-aligned points (2 per axis, +/- 2.0)
        axis_step = 2.0
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * axis_step
                init_points.append(x)
        
        # 3. Diagonal points in first 2 dims (4 points)
        for sx in [1, -1]:
            for sy in [1, -1]:
                x = np.zeros(dim)
                x[0] = sx * axis_step
                x[1] = sy * axis_step
                init_points.append(x)
                
        # 4. Random points to fill gaps
        # We have used 1 + 2*5 + 4 = 16 points.
        # Budget is 500. We can afford more. Let's add 8 random points.
        for _ in range(8):
            x = rng.uniform(-4.5, 4.5, dim)
            init_points.append(x)
            
        # Evaluate all init points
        for x in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = call_func(x)
            
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead
        start_x = best_x.copy()
        
        # Initialize simplex
        simplex = [start_x.copy()]
        simplex_f = [best_f]
        
        # Step size for simplex initialization
        step_size = 0.8
        for i in range(dim):
            v = start_x.copy()
            v[i] += step_size
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)
            
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 150 # Limit iterations to leave budget for polish
        tol = 1e-10
        no_improve_count = 0
        best_f_nm = best_f
        
        for iteration in range(max_iter):
            if calls_made >= budget - 50 or time.time() > deadline - 10: # Leave 50 for polish
                break
            
            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break
            
            # Track no-improvement
            if simplex_f[0] < best_f_nm - 1e-15:
                no_improve_count = 0
                best_f_nm = simplex_f[0]
            else:
                no_improve_count += 1
                
            # Restart logic: if stuck, try a random point near best
            if no_improve_count >= 20 and calls_made < budget - 50:
                # Generate a new point near current best
                x_new = simplex[0] + rng.normal(0, 0.5, dim)
                x_new = np.clip(x_new, -4.5, 4.5)
                f_new = call_func(x_new)
                if f_new < simplex_f[0]:
                    simplex = [x_new.copy()]
                    simplex_f = [f_new]
                    for i in range(dim):
                        v = x_new.copy()
                        v[i] += 0.5
                        fv = call_func(v)
                        simplex.append(v.copy())
                        simplex_f.append(fv)
                    no_improve_count = 0
                    best_f_nm = f_new
                    continue
                else:
                    # If restart failed, reduce step size or break?
                    # Let's just continue, but maybe shrink simplex more aggressively
                    pass

            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])
                        
        # Phase 3: Adaptive Gradient-Scaled Local Search
        # Use remaining budget to refine using gradient magnitude for step sizing.
        
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline - 5:
            # Estimate gradient magnitude at current best
            # Use central differences for better accuracy
            h = 1e-4
            grad = np.zeros(dim)
            
            # We need to be careful with budget here.
            # Let's do a few iterations of gradient estimation and update.
            
            # Initial gradient estimate
            for i in range(dim):
                if calls_made >= budget - 10 or time.time() > deadline - 5:
                    break
                x_plus = best_x.copy()
                x_plus[i] += h
                x_minus = best_x.copy()
                x_minus[i] -= h
                f_plus = func(np.clip(x_plus, -5, 5))
                f_minus = func(np.clip(x_minus, -5, 5))
                calls_made += 2
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Iterative refinement
            # Step size is scaled by inverse of gradient magnitude
            # If gradient is large, step small. If gradient is small, step larger.
            # But we also need to handle the case where gradient is zero (at optimum).
            
            base_step = 1.0
            min_step = 1e-6
            max_step = 1.0
            
            for _ in range(50): # Max 50 refinement steps
                if calls_made >= budget - 10 or time.time() > deadline - 5:
                    break
                
                # Estimate gradient magnitude
                # We already have grad from above, but it might be stale.
                # Let's re-estimate if we've moved.
                # For efficiency, we can use the last estimated grad and update it.
                
                grad_norm = np.linalg.norm(grad)
                
                if grad_norm < 1e-8:
                    # Gradient is effectively zero, we are likely at optimum
                    # Do a small random perturbation to check if we stuck in flat region
                    step = base_step * 0.1
                    x_try = best_x + rng.normal(0, step, dim)
                    x_try = np.clip(x_try, -4.5, 4.5)
                    f_try = call_func(x_try)
                    if f_try < best_f:
                        # Move and re-estimate gradient
                        # Re-estimate gradient at new point
                        for i in range(dim):
                            if calls_made >= budget - 10 or time.time() > deadline - 5:
                                break
                            x_plus = best_x.copy()
                            x_plus[i] += h
                            x_minus = best_x.copy()
                            x_minus[i] -= h
                            f_plus = func(np.clip(x_plus, -5, 5))
                            f_minus = func(np.clip(x_minus, -5, 5))
                            calls_made += 2
                            grad[i] = (f_plus - f_minus) / (2 * h)
                    else:
                        # No improvement, shrink step
                        base_step *= 0.5
                        if base_step < min_step:
                            break
                    continue
                
                # Scale step by inverse gradient magnitude
                # Use a damping factor to avoid huge steps
                step_size = base_step / (1.0 + grad_norm)
                step_size = np.clip(step_size, min_step, max_step)
                
                # Move in negative gradient direction
                x_try = best_x - step_size * (grad / grad_norm)
                x_try = np.clip(x_try, -4.5, 4.5)
                f_try = call_func(x_try)
                
                if f_try < best_f:
                    # Move
                    # Re-estimate gradient at new point
                    for i in range(dim):
                        if calls_made >= budget - 10 or time.time() > deadline - 5:
                            break
                        x_plus = best_x.copy()
                        x_plus[i] += h
                        x_minus = best_x.copy()
                        x_minus[i] -= h
                        f_plus = func(np.clip(x_plus, -5, 5))
                        f_minus = func(np.clip(x_minus, -5, 5))
                        calls_made += 2
                        grad[i] = (f_plus - f_minus) / (2 * h)
                    
                    # Increase base step slightly if we made good progress
                    base_step = min(base_step * 1.1, max_step)
                else:
                    # No improvement, shrink step
                    base_step *= 0.5
                    if base_step < min_step:
                        break

        return best_x, best_f
