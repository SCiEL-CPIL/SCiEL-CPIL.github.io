import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Phase 1: (1+10)-ES for global exploration
        # Budget split: 80% for ES, 20% for NM polish
        es_budget = int(self.budget * 0.8)
        nm_budget = self.budget - es_budget
        
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        
        # (1+10)-ES parameters
        lambda_ = 10
        # We evaluate lambda_ candidates per step.
        # Each step consumes lambda_ calls.
        # Total steps approx es_budget / lambda_
        
        while calls_made + lambda_ <= es_budget:
            if time.time() > deadline - 10:
                break
                
            # Generate lambda_ candidates
            candidates = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                ft = func(x_trial)
                calls_made += 1
                candidates.append((ft, x_trial))
            
            # Sort candidates
            candidates.sort(key=lambda t: t[0])
            
            # Update best
            if candidates[0][0] < best_fx:
                best_fx = candidates[0][0]
                best_x = candidates[0][1].copy()
            
            # Adaptation: If the best candidate is better than the parent, expand sigma
            # Otherwise, shrink sigma
            if candidates[0][0] < fx:
                x = candidates[0][1].copy()
                fx = candidates[0][0]
                sigma *= 1.1
            else:
                # Move x towards the best candidate to stay in the basin, but shrink sigma
                # Or just keep x and shrink sigma?
                # Standard (1+1) ES keeps x if no improvement. 
                # But to explore, maybe move towards best?
                # Let's stick to standard (1+1) logic for stability: x stays, sigma shrinks.
                sigma *= 0.9
                
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)

        # Phase 2: Nelder-Mead Polish
        # Use remaining budget
        if calls_made < self.budget and time.time() < deadline - 5:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            for i in range(dim):
                v = best_x.copy()
                v[i] += 0.1 if v[i] < 4.9 else -0.1
                simplex.append(v)
            
            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if calls_made >= self.budget:
                    break
                fv = func(v)
                calls_made += 1
                simplex_f.append(fv)
            
            if len(simplex_f) < dim + 1:
                # Not enough budget to even start NM properly, skip or do minimal
                pass
            else:
                # Run NM
                max_nm_calls = self.budget - calls_made
                # Standard NM loop
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 2.0
                
                nm_calls_used = 0
                while nm_calls_used < max_nm_calls:
                    if time.time() > deadline - 2:
                        break
                    
                    # Reflect
                    centroid = np.mean(simplex[:-1], axis=0)
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    nm_calls_used += 1
                    
                    if f_r < simplex_f[0]:
                        # Expand
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if nm_calls_used + 1 > max_nm_calls:
                            break
                        f_e = func(x_e)
                        nm_calls_used += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contract
                        if f_r < simplex_f[-1]:
                            # Inner contract
                            x_c = centroid + 0.5 * (x_r - centroid)
                        else:
                            # Outer contract
                            x_c = centroid - 0.5 * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        if nm_calls_used + 1 > max_nm_calls:
                            break
                        f_c = func(x_c)
                        nm_calls_used += 1
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if nm_calls_used + 1 > max_nm_calls:
                                    break
                                simplex_f[i] = func(simplex[i])
                                nm_calls_used += 1
                            if nm_calls_used >= max_nm_calls:
                                break
                    
                    # Update best
                    current_best_idx = np.argmin(simplex_f)
                    if simplex_f[current_best_idx] < best_fx:
                        best_fx = simplex_f[current_best_idx]
                        best_x = simplex[current_best_idx].copy()
                    
                    # Re-sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

        return best_x, best_fx
