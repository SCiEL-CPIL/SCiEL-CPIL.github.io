import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Strategy:
        # We will run multiple Nelder-Mead instances.
        # Each instance gets a limited budget.
        # If an instance improves the global best, we restart near the best.
        # If it fails, we restart randomly.
        
        max_evals_per_nm = 25
        max_restarts = 10
        
        for restart in range(max_restarts):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Determine start point
            if restart == 0:
                x_start = x_best
            else:
                # 50% chance to explore globally, 50% to exploit locally
                if rng.random() < 0.5:
                    x_start = rng.uniform(-5, 5, self.dim)
                else:
                    # Local perturbation around best
                    # Scale noise based on how good the best is? 
                    # Just use a fixed reasonable step for 5D box
                    noise = rng.standard_normal(self.dim) * 1.0
                    x_start = np.clip(x_best + noise, -5, 5)
            
            # Initialize Simplex
            # Simplex: x_start and x_start + step * e_i
            step = 0.5
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x_start
            f_values = np.zeros(self.dim + 1)
            f_values[0] = func(x_start)
            calls += 1
            
            for i in range(self.dim):
                x_i = x_start.copy()
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                simplex[i+1] = x_i
                f_values[i+1] = func(x_i)
                calls += 1
                if calls >= self.budget:
                    break
            
            # If we broke due to budget, check best and exit
            if calls >= self.budget:
                # Update global best with any point in simplex
                idx_min = np.argmin(f_values)
                if f_values[idx_min] < f_best:
                    f_best = f_values[idx_min]
                    x_best = simplex[idx_min].copy()
                break
            
            # Nelder-Mead Loop
            nm_evals = 0
            max_nm_evals = max_evals_per_nm
            stagnant_count = 0
            last_best = f_best
            
            while nm_evals < max_nm_evals and calls < self.budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check stagnation
                if f_values[0] < last_best:
                    last_best = f_values[0]
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                    if stagnant_count > 5:
                        break
                
                # Centroid of best dim points (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                nm_evals += 1
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    nm_evals += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outer contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inner contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    nm_evals += 1
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            calls += 1
                            nm_evals += 1
                            if calls >= self.budget:
                                break
                        if calls >= self.budget:
                            break
                
                # Update global best
                idx_min = np.argmin(f_values)
                if f_values[idx_min] < f_best:
                    f_best = f_values[idx_min]
                    x_best = simplex[idx_min].copy()
            
            # If budget exhausted inside NM, break outer loop
            if calls >= self.budget:
                break

        return x_best, f_best
