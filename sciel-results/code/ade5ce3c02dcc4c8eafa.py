import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Parameters
        lam = 10  # Population size
        mu = 5    # Number of parents
        sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 10.0
        
        # State
        best_x = None
        best_f = np.inf
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (lam, d))
        evals = 0
        pop_f = np.array([func(x) for x in pop])
        evals += lam
        
        # Update best
        idx_min = np.argmin(pop_f)
        best_x = pop[idx_min].copy()
        best_f = pop_f[idx_min]
        
        no_improve_count = 0
        
        while evals < budget:
            # Check if we can fit a full generation
            if evals + lam > budget:
                # Fallback to single-step search to use remaining budget safely
                while evals < budget:
                    y = best_x + sigma * np.random.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                break
            
            # Selection: pick best mu
            idx_sort = np.argsort(pop_f)
            parents = pop[idx_sort[:mu]]
            
            # Offspring generation
            offspring = []
            for i in range(mu):
                for j in range(lam // mu):
                    # Gaussian perturbation
                    step = sigma * np.random.standard_normal(d)
                    y = parents[i] + step
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
            
            offspring = np.array(offspring)
            
            # Evaluate offspring
            off_f = np.array([func(x) for x in offspring])
            evals += lam
            
            # Update global best
            idx_off_min = np.argmin(off_f)
            if off_f[idx_off_min] < best_f:
                best_f = off_f[idx_off_min]
                best_x = offspring[idx_off_min].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Combine and select
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, off_f])
            
            idx_select = np.argsort(combined_f)[:lam]
            pop = combined[idx_select]
            pop_f = combined_f[idx_select]
            
            # Adapt sigma
            if no_improve_count == 0:
                sigma *= 1.2
            else:
                sigma *= 0.9
            
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Stagnation Restart
            if no_improve_count > 10:
                # Restart population
                pop = np.random.uniform(-5, 5, (lam, d))
                pop[0] = best_x
                pop_f = np.array([func(x) for x in pop])
                evals += lam
                
                # Update best if the new pop has something better (unlikely, but safe)
                idx_new_min = np.argmin(pop_f)
                if pop_f[idx_new_min] < best_f:
                    best_f = pop_f[idx_new_min]
                    best_x = pop[idx_new_min].copy()
                
                # Reset sigma
                sigma = 2.0
                no_improve_count = 0
                
                # Check budget again
                if evals >= budget:
                    break
        
        return best_x, best_f
