"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Nelder-Mead simplex optimizer with ridge-aware adaptation for BBOB f13.
    
    Strategy:
    1. Initialize simplex with diverse points.
    2. Run Nelder-Mead with adaptive step sizes and restarts.
    3. When simplex collapses (indicating we're on the ridge), switch to 
       1D line search along the ridge direction for faster convergence.
    4. Hybrid approach: use NM for navigation, line search for exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Initialize Nelder-Mead simplex
        # Start with origin and spread out
        simplex = []
        fvals = []
        
        # First point at origin
        x0 = np.zeros(n)
        f0 = wrapped_func(x0)
        simplex.append(x0.copy())
        fvals.append(f0)
        
        # Other points spread along axes
        base_step = 1.0
        for i in range(n):
            xi = x0.copy()
            xi[i] = base_step
            fi = wrapped_func(xi)
            simplex.append(xi.copy())
            fvals.append(fi)
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink
        
        # Track simplex size for ridge detection
        nm_iterations = 0
        max_nm_iterations = 100
        restart_count = 0
        max_restarts = 5
        
        # Phase 1: Nelder-Mead until it starts converging or budget runs low
        while calls_made[0] < budget * 0.6 and time.time() < deadline:
            nm_iterations += 1
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check if simplex is collapsed (ridge detection)
            centroid = np.mean(simplex[:-1], axis=0)
            simplex_size = np.linalg.norm(simplex[-1] - centroid)
            
            # If simplex is very small, we might be on the ridge
            if simplex_size < 0.01:
                break
            
            # Standard Nelder-Mead steps
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = wrapped_func(x_ref)
            
            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = wrapped_func(x_exp)
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                
                x_con = np.clip(x_con, -5, 5)
                f_con = wrapped_func(x_con)
                
                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        x_shrunk = np.clip(simplex[i], -5, 5)
                        f_shrunk = wrapped_func(x_shrunk)
                        simplex[i] = x_shrunk
                        fvals[i] = f_shrunk
                    
                    restart_count += 1
                    if restart_count >= max_restarts:
                        break
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
        
        # Phase 2: Ridge exploitation with 1D line search
        # Estimate ridge direction from the simplex or use gradient-based approach
        remaining_budget = budget - calls_made[0]
        
        if remaining_budget > 10:
            # Try to estimate the ridge direction
            # Use finite differences to get gradient
            h = 0.01
            grad = np.zeros(n)
            x_center = best_x.copy()
            
            for i in range(n):
                if calls_made[0] >= budget or time.time() >= deadline:
                    break
                x_plus = x_center.copy()
                x_minus = x_center.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = wrapped_func(x_plus)
                f_minus = wrapped_func(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # If gradient is small, we're near minimum
            # Otherwise, search along negative gradient direction
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm > 1e-6:
                # Line search along -gradient direction
                direction = -grad / grad_norm
                
                # Adaptive line search
                step_size = 0.1
                x_cur = best_x.copy()
                f_cur = best_f
                
                for _ in range(remaining_budget // 2):
                    if calls_made[0] >= budget or time.time() >= deadline:
                        break
                    
                    # Try step
                    x_try = x_cur + step_size * direction
                    x_try = np.clip(x_try, -5, 5)
                    f_try = wrapped_func(x_try)
                    
                    if f_try < f_cur:
                        x_cur = x_try
                        f_cur = f_try
                        step_size = min(step_size * 1.2, 1.0)
                    else:
                        step_size = max(step_size * 0.5, 1e-6)
                        
                        # If step is too small, try different direction
                        if step_size < 1e-5:
                            # Perturb direction
                            direction = direction + 0.1 * rng.standard_normal(n)
                            direction /= np.linalg.norm(direction)
                            step_size = 0.05
                    
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
            
            # Phase 3: Final fine-tuning with small ES
            final_budget = budget - calls_made[0]
            if final_budget > 5:
                x_es = best_x.copy()
                fx_es = best_f
                sigma_es = 0.01
                
                for _ in range(final_budget):
                    if calls_made[0] >= budget or time.time() >= deadline:
                        break
                    
                    y = np.clip(x_es + sigma_es * rng.standard_normal(n), -5, 5)
                    fy = wrapped_func(y)
                    
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma_es = min(sigma_es * 1.05, 0.05)
                    else:
                        sigma_es = max(sigma_es * 0.95, 1e-6)
                
                if fx_es < best_f:
                    best_f = fx_es
                    best_x = x_es.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
