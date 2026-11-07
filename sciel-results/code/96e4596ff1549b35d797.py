"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Differential Evolution + Nelder-Mead polish for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def _make_func(self, func):
        """Wrap func to count calls."""
        def wrapped(x):
            self.call_count += 1
            return float(func(x))
        return wrapped

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.call_count = 0
        
        # Bounds
        bounds = [(-5.0, 5.0)] * dim
        
        # Create a counting wrapper
        wrapped_func = self._make_func(func)
        
        # Phase 1: Differential Evolution for global exploration
        # DE is robust to local minima and good for multimodal functions like Lunacek bi-Rastrigin
        # Use a smaller population and maxiter to conserve budget
        # maxiter is not directly budget, but we can control via population and iterations
        # Let's use a fixed number of DE iterations that roughly uses half the budget
        # Population size = 10 * dim = 50
        # Each iteration = population size evaluations = 50
        # Let's use ~40 iterations -> 2000 evals? No, budget is 500.
        # So ~10 iterations * 50 = 500.
        # But we want to save some for local polishing.
        # Let's use population size = 5 * dim = 25
        # 15 iterations * 25 = 375 evals
        # Remaining ~125 for Nelder-Mead
        
        pop_size = 5 * dim
        de_maxiter = 15
        # DE initial population uses pop_size evals
        # Then each generation uses pop_size evals
        # Total DE evals = pop_size * (de_maxiter + 1)
        # 25 * 16 = 400
        
        # We need to be careful with the call count.
        # scipy's differential_evolution does not expose a callback for every eval easily in older versions,
        # but it does accept a `callback` that is called after each generation.
        # However, we need to stop if we exceed budget.
        # Since scipy's DE is a black box, it's hard to interrupt mid-generation.
        # Instead, we can implement a simple DE ourselves or use a conservative budget.
        
        # Let's implement a simple DE ourselves to have full control over budget.
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = rng.uniform(-5.0, 5.0, size=(pop_size, dim))
        fit = np.array([wrapped_func(ind) for ind in pop])
        
        # Check if we already exceeded budget (shouldn't happen with 25*1=25)
        if self.call_count >= budget:
            best_idx = np.argmin(fit)
            return pop[best_idx], fit[best_idx]
        
        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        for gen in range(de_maxiter):
            # Stop if we are close to budget
            if self.call_count + pop_size > budget:
                break
            
            best_idx = np.argmin(fit)
            best_x = pop[best_idx]
            best_f = fit[best_idx]
            
            for i in range(pop_size):
                # Choose three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty_like(pop[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluation
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
            
            # Check budget
            if self.call_count >= budget:
                break
        
        # Find best from DE
        best_idx = np.argmin(fit)
        x_best = pop[best_idx].copy()
        f_best = fit[best_idx]
        
        # Phase 2: Nelder-Mead local refinement
        # Use remaining budget
        remaining_budget = budget - self.call_count
        if remaining_budget > 10:
            # Nelder-Mead options
            # maxiter is the max number of iterations, not function calls directly
            # Each iteration can use up to (dim+1) function calls
            # Let's set maxiter to a reasonable number
            nm_maxiter = max(1, remaining_budget // (dim + 1))
            
            try:
                res = minimize(
                    wrapped_func,
                    x_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'initial_simplex': None,
                        'adaptive': True
                    }
                )
                if res.fun < f_best:
                    x_best = res.x
                    f_best = res.fun
            except Exception:
                pass
        
        return x_best, f_best
# EVOLVE-BLOCK-END
