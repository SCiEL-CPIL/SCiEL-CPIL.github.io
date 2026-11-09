import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # 1. Initial Exploration: Sample a few points to find a good starting region
        # Discus functions can have local minima. A few random starts help.
        # Let's use 10% of the budget for initial scouting.
        scout_budget = max(10, self.budget // 10)
        
        # Generate scout points
        # Use a fixed seed for reproducibility in tests, but the task implies
        # the evaluator handles the specific instance. The prompt says "hidden seeded shift".
        # The Optimizer class is instantiated per instance. We should use a random seed 
        # or a fixed one? The seed program uses default_rng(0). 
        # If we use a fixed seed, the scout points are the same for all instances.
        # This is fine as the landscape is shifted.
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_evals = 0
        
        # Scout phase
        scout_points = rng.uniform(-5, 5, (scout_budget, self.dim))
        for p in scout_points:
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
            fp = func(p)
            used_evals += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # If we didn't use all scout budget, we can just proceed.
        
        # 2. Local Optimization: Nelder-Mead
        # We will run Nelder-Mead from the best scout point.
        # We might also run it from the 2nd best if we have budget, but 
        # focusing on the best is usually better for AOCC (early progress).
        
        # Initialize Simplex
        # Standard Nelder-Mead initialization
        x0 = best_x
        f0 = best_f
        
        # Simplex vertices
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        # Add other vertices with small perturbations
        for i in range(self.dim):
            x_new = x0.copy()
            # Perturb dimension i
            # Step size relative to box size
            step = 0.5 if abs(x0[i]) < 4.5 else 0.1
            if x0[i] + step > 5:
                x_new[i] = x0[i] - step
            else:
                x_new[i] = x0[i] + step
            simplex.append(x_new)
            if used_evals < self.budget and time.time() < deadline:
                simplex_f.append(func(x_new))
                used_evals += 1
            else:
                simplex_f.append(np.inf)
                break
        
        # If we ran out of budget during init, return best
        if used_evals >= self.budget:
            return best_x, best_f
            
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Main Loop
        while used_evals < self.budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            if len(simplex_f) > 1:
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10 * max(1, abs(simplex_f[0])):
                    break
            
            # Compute centroid (excluding worst point)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            f_r = func(x_r)
            used_evals += 1
            
            if f_r < simplex_f[0]:
                # Try Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                
                if used_evals < self.budget and time.time() < deadline:
                    f_e = func(x_e)
                    used_evals += 1
                    if f_e < f_r:
                        x_new, f_new = x_e, f_e
                    else:
                        x_new, f_new = x_r, f_r
                else:
                    x_new, f_new = x_r, f_r
            elif f_r < simplex_f[-2]:
                x_new, f_new = x_r, f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                f_c = func(x_c)
                used_evals += 1
                
                if f_c < min(f_r, simplex_f[-1]):
                    x_new, f_new = x_c, f_c
                else:
                    # Shrink
                    x_new_list = []
                    f_new_list = []
                    for i in range(1, len(simplex)):
                        x_sh = simplex[0] + sigma * (simplex[i] - simplex[0])
                        x_sh = np.clip(x_sh, -5, 5)
                        x_new_list.append(x_sh)
                        if used_evals < self.budget and time.time() < deadline:
                            f_sh = func(x_sh)
                            used_evals += 1
                            f_new_list.append(f_sh)
                        else:
                            f_new_list.append(simplex_f[i])
                    
                    simplex = x_new_list
                    simplex_f = f_new_list
                    continue # Restart loop with new simplex
            
            # Update simplex: Replace worst point with new best
            simplex[-1] = x_new
            simplex_f[-1] = f_new
            
            # Update global best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        
        return best_x, best_f
