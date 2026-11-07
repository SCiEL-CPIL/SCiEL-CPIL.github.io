import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.elapsed_time = 0.0
        self.max_time = 300.0 # Safety margin for 330s kill

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        # 1. Initialize diverse population
        # Use Latin Hypercube for better coverage than pure random
        N = 10 # Population size
        x_pop = np.zeros((N, self.dim))
        for i in range(N):
            x_pop[i] = rng.uniform(-5, 5, self.dim)
            
        f_pop = np.array([func(x) for x in x_pop])
        best_idx = np.argmin(f_pop)
        x_best = x_pop[best_idx].copy()
        f_best = f_pop[best_idx]
        
        calls_used = N
        
        # 2. Differential Evolution Global Search
        # Budget split: 60% for DE, 40% for Local
        de_budget = int(self.budget * 0.6)
        local_budget = self.budget - de_budget
        
        # DE Parameters
        F = 0.8
        CR = 0.5
        
        while calls_used < de_budget:
            # Check time
            if time.time() - self.start_time > self.max_time * 0.9:
                break
                
            for i in range(N):
                if calls_used >= de_budget:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                # Ensure they are not i (standard DE usually allows, but distinct is better)
                # Actually, standard DE/rand/1 picks 3 distinct from the population.
                
                # Mutation
                v = x_pop[r1] + F * (x_pop[r2] - x_pop[r3])
                
                # Crossover
                j_rand = rng.integers(self.dim)
                trial = np.empty_like(x_pop[i])
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = x_pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_used += 1
                
                # Selection
                if f_trial <= f_pop[i]:
                    x_pop[i] = trial
                    f_pop[i] = f_trial
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = trial.copy()
                        
        # 3. Local Polish with L-BFGS-B or Nelder-Mead
        # We have local_budget calls left.
        # L-BFGS-B uses finite differences, so each "call" to the optimizer function
        # might use multiple function evaluations.
        # We need to be careful with the callback to count calls.
        
        # Let's use a wrapper to count calls for the local optimizer
        local_calls = [0]
        def local_func(x):
            local_calls[0] += 1
            return func(np.clip(x, -5, 5))
            
        # Try L-BFGS-B first, it's fast for smooth functions.
        # If it fails or uses too many, fall back to Nelder-Mead.
        
        # We can't easily limit L-BFGS-B to exactly local_budget calls without a custom callback that raises an exception,
        # which might be messy. 
        # Instead, we can run a few steps of Nelder-Mead which is simpler to control?
        # Or just run L-BFGS-B with a maxiter that corresponds to roughly the budget.
        # In 5D, L-BFGS-B typically needs ~10-20 iterations to converge, each taking ~5-10 evals.
        # So 100 evals is plenty for 1-2 restarts or one good run.
        
        try:
            # Reseed the local search from the global best
            res = minimize(local_func, x_best, method='L-BFGS-B', 
                           options={'maxiter': local_budget // 10, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < f_best:
                f_best = res.fun
                x_best = res.x
        except Exception:
            pass

        # If we still have budget and time, try Nelder-Mead from the current best
        remaining = self.budget - calls_used - local_calls[0]
        if remaining > 10 and time.time() - self.start_time < self.max_time * 0.95:
            local_calls[0] = 0 # Reset counter for this phase
            try:
                res2 = minimize(local_func, x_best, method='Nelder-Mead', 
                                options={'maxiter': remaining // 5, 'fatol': 1e-12, 'xatol': 1e-12})
                if res2.fun < f_best:
                    f_best = res2.fun
                    x_best = res2.x
            except Exception:
                pass

        return x_best, f_best
