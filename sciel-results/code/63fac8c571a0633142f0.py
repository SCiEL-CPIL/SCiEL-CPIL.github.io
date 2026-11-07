import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290
        
        call_count = 0
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            return func(x)

        rng = np.random.default_rng(42)
        
        # Phase 1: (mu+lambda)-ES
        # mu=5, lambda=20
        # Budget for ES: 200 calls
        # 10 generations * 20 offspring = 200 calls
        # Initial population: 20 calls
        # Total ES: 220 calls
        
        es_gen_budget = 10
        
        # Initialize population
        pop = rng.uniform(-5, 5, (20, self.dim))
        pop_f = np.array([wrapped_func(x) for x in pop])
        
        best_x = pop[np.argmin(pop_f)]
        best_f = np.min(pop_f)
        
        sigma = 1.0
        
        for gen in range(es_gen_budget):
            if time.time() > deadline - 60:
                break
                
            # Generate offspring
            parents_idx = np.argsort(pop_f)[:5]
            parents = pop[parents_idx]
            
            offspring = []
            for _ in range(20):
                p = parents[rng.integers(0, 5)]
                step = sigma * rng.standard_normal(self.dim)
                child = np.clip(p + step, -5, 5)
                offspring.append(child)
            
            offspring = np.array(offspring)
            try:
                offspring_f = np.array([wrapped_func(x) for x in offspring])
            except RuntimeError:
                break
            
            # Selection
            combined = np.vstack([pop, offspring])
            combined_f = np.concatenate([pop_f, offspring_f])
            
            idx_best = np.argmin(combined_f)
            if combined_f[idx_best] < best_f:
                best_f = combined_f[idx_best]
                best_x = combined[idx_best]
            
            idx_sort = np.argsort(combined_f)[:20]
            pop = combined[idx_sort]
            pop_f = combined_f[idx_sort]
            
            # Adapt sigma
            # If the best of the offspring is better than the best of the parents, increase sigma
            # Otherwise decrease
            best_offspring = np.min(offspring_f)
            best_parent = np.min(pop_f) # Note: pop_f is updated after this block, so use old best? 
            # Actually, best_parent is the min of the *current* pop before replacement.
            # Let's just use a simple rule: if global best improved, increase, else decrease.
            if best_f < (pop_f[0] if pop_f[0] != best_f else pop_f[1] if len(pop_f)>1 else best_f):
                 # This logic is messy. Let's simplify.
                 pass
            
            # Simpler sigma adaptation:
            # Count successes in the last generation
            # If > 50% of offspring were better than their parents (approx), increase
            # For now, just keep sigma constant or decay slowly.
            sigma *= 0.95
            sigma = np.clip(sigma, 0.5, 3.0)

        # Phase 2: Local refinement with L-BFGS-B
        remaining_budget = self.budget - call_count
        if remaining_budget > 10:
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': remaining_budget // 2, 'maxfun': remaining_budget}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
