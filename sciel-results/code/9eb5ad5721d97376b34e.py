import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_used = 0
        n = self.dim
        
        # Phase 1: (1+1)-ES with 1/5 success rule
        x = np.zeros(n)
        fx = func(x)
        calls_used += 1
        sigma = 1.0
        
        # Use 70% of budget for ES
        es_budget = int(self.budget * 0.7)
        
        success_count = 0
        window_size = 10
        window = [False] * window_size
        window_idx = 0
        
        while calls_used < es_budget:
            if time.time() > deadline:
                break
            
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                window[window_idx] = True
            else:
                window[window_idx] = False
                
            window_idx = (window_idx + 1) % window_size
            
            # 1/5 success rule: if >1/5 of last window succeeded, increase sigma
            if success_count > window_size / 5:
                sigma *= 1.5
            elif success_count < window_size / 5:
                sigma *= 0.5
                
            if sigma < 1e-8:
                break

        # Phase 2: Nelder-Mead Polish
        # If we have time and budget left
        if calls_used < self.budget and time.time() < deadline:
            x, fx = self._nelder_mead(func, x, fx, self.budget - calls_used, deadline)
            
        return x, fx

    def _nelder_mead(self, func, x0, fx0, budget, deadline):
        n = self.dim
        # Initialize simplex
        simplex = [x0]
        f_values = [fx0]
        calls_made = 1
        
        # Create other vertices with larger perturbations for better coverage
        for i in range(n):
            x_new = x0.copy()
            x_new[i] += 0.5  # Larger perturbation
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            simplex.append(x_new)
            f_values.append(f_new)
            
        # Update best
        best_idx = np.argmin(f_values)
        x_best = simplex[best_idx]
        f_best = f_values[best_idx]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex
            idx = np.argsort(f_values)
            simplex = [simplex[i] for i in idx]
            f_values = [f_values[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_made += 1
            
            if f_ref < f_values[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_made += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
            elif f_ref < f_values[-2]:
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls_made += 1
                if f_con < f_values[-1]:
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= budget:
                            break
                    if calls_made >= budget:
                        break

            # Update best
            best_idx = np.argmin(f_values)
            if f_values[best_idx] < f_best:
                f_best = f_values[best_idx]
                x_best = simplex[best_idx]

        return x_best, f_best
