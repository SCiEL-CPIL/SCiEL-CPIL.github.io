"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start DE + Nelder-Mead optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        
        # Track best
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            if calls >= budget:
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution with population
        # Use a moderate population size
        pop_size = 20
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        
        # Initialize population randomly in [-5, 5]^n
        pop = rng.uniform(-5, 5, (pop_size, n))
        fvals = np.array([safe_func(x) for x in pop])
        
        # Sort by fitness
        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]
        
        # DE evolution
        max_gens = 30
        for gen in range(max_gens):
            if calls >= budget:
                break
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            new_pop = []
            new_fvals = []
            
            for i in range(pop_size):
                if calls >= budget:
                    new_pop.append(pop[i])
                    new_fvals.append(fvals[i])
                    continue
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == r2 or r2 == r3 or r1 == r3:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binomial crossover
                j_rand = rng.randint(n)
                trial = np.empty(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Bound
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = safe_func(trial)
                
                # Selection
                if ft <= fvals[i]:
                    new_pop.append(trial)
                    new_fvals.append(ft)
                else:
                    new_pop.append(pop[i])
                    new_fvals.append(fvals[i])
            
            pop = np.array(new_pop)
            fvals = np.array(new_fvals)
            
            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
        
        # Phase 2: Local refinement with Nelder-Mead on the best point
        remaining = budget - calls
        if remaining > 10:
            # Use Nelder-Mead from scipy
            from scipy.optimize import minimize
            
            # Set up bounds
            bounds = [(-5, 5)] * n
            
            # Wrap func for scipy
            def scipy_func(x):
                nonlocal calls
                if calls >= budget:
                    return 1e10
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            
            # Nelder-Mead parameters
            try:
                res = minimize(
                    scipy_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': max(1, remaining // 3), 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
