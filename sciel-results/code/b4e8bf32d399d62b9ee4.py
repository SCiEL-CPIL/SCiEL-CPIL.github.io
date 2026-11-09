"""Optimizer for BBOB f23 (Katsuura) using Multi-Start (1+1)-ES with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with LHS initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        n_lhs = 30
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        # Phase 2: Multi-Start Local Search
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Reserve budget for final polishing phase (Nelder-Mead + micro-polish)
        reserve_polish = min(40, max(20, remaining_budget // 4))
        search_budget = remaining_budget - reserve_polish
        
        if search_budget <= 0:
            search_budget = remaining_budget
            reserve_polish = 0

        # Define trial structure
        trial_length = 70
        n_trials = search_budget // trial_length
        
        if n_trials == 0:
            trial_length = search_budget
            n_trials = 1

        sigma_init = 1.5
        min_sigma = 0.0005  # Lower min_sigma for finer convergence
        max_sigma = 4.0
        jump_scale = 2.5

        for t in range(n_trials):
            if calls_made >= budget - reserve_polish:
                break
            if time.time() > deadline:
                return x_global_best, fx_global_best
            
            if t == 0:
                x = x_global_best.copy()
            else:
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            
            for step in range(trial_length - 1):
                if calls_made >= budget - reserve_polish:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma)
                    update_best(x, fx)
                else:
                    sigma = max(sigma * 0.8, min_sigma)

        # Phase 3: Nelder-Mead Polish
        remaining = budget - calls_made
        if remaining > 10 and x_global_best is not None:
            # Build initial simplex around global best
            simplex = [x_global_best.copy()]
            f_simplex = [fx_global_best]
            
            step_size = 0.1
            for i in range(dim):
                perturbation = x_global_best.copy()
                if perturbation[i] > upper[i] - step_size:
                    perturbation[i] -= step_size
                else:
                    perturbation[i] += step_size
                perturbation = np.clip(perturbation, bounds, upper)
                fx_pert = func(perturbation)
                calls_made += 1
                simplex.append(perturbation)
                f_simplex.append(fx_pert)
                update_best(perturbation, fx_pert)
                
                if calls_made >= budget - 5 or time.time() > deadline:
                    return x_global_best, fx_global_best
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            # Run Nelder-Mead until budget is nearly exhausted
            while calls_made < budget - 5 and time.time() < deadline:
                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Convergence check
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, bounds, upper)
                fx_r = func(x_r)
                calls_made += 1
                update_best(x_r, fx_r)
                
                if fx_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, bounds, upper)
                    fx_e = func(x_e)
                    calls_made += 1
                    update_best(x_e, fx_e)
                    
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = fx_r
                elif fx_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < f_simplex[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, bounds, upper)
                    fx_c = func(x_c)
                    calls_made += 1
                    update_best(x_c, fx_c)
                    
                    if fx_c < min(fx_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = fx_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds, upper)
                            if calls_made >= budget - 2 or time.time() > deadline:
                                return x_global_best, fx_global_best
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            update_best(simplex[i], f_simplex[i])
                
                if calls_made >= budget - 2 or time.time() > deadline:
                    return x_global_best, fx_global_best

        # Phase 4: Micro-polish with very small steps
        remaining = budget - calls_made
        if remaining > 5 and x_global_best is not None:
            x = x_global_best.copy()
            fx = fx_global_best
            micro_sigma = 0.001
            for _ in range(remaining):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = x + micro_sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                # Slowly decay sigma
                micro_sigma *= 0.95

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
