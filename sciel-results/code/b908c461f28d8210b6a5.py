import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return 1e10
            x = clip(x)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Dense random initialization for early progress
        n_init = 25
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-scale differential evolution with adaptive crossover
        # Three populations: large (global), medium (basin), small (fine-tune)
        
        def run_de(start_x, n_evals, sigma_scale, crossover_rate):
            nonlocal calls, best_f, best_x, deadline, budget
            if calls >= budget or time.time() > deadline:
                return
            
            # Initialize population around start_x
            lam = 12  # population size
            if n_evals < lam:
                lam = max(4, n_evals // 2)
            
            pop = [start_x + rng.normal(0, sigma_scale, dim) for _ in range(lam)]
            pop_f = []
            for x in pop:
                if calls >= budget or time.time() > deadline:
                    return
                pop_f.append(evaluate(x))
            
            # Differential evolution loop
            while calls < budget and time.time() < deadline:
                # Sort population
                sorted_idx = np.argsort(pop_f)
                pop = [pop[i] for i in sorted_idx]
                pop_f = [pop_f[i] for i in sorted_idx]
                
                # Generate offspring
                new_pop = []
                new_pop_f = []
                
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Differential mutation
                    r1, r2, r3 = rng.choice(lam, 3, replace=False)
                    while r1 == r2 or r1 == r3 or r2 == r3:
                        r1, r2, r3 = rng.choice(lam, 3, replace=False)
                    
                    F = 0.5 + rng.random() * 0.5  # scale factor
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover
                    trial = np.zeros(dim)
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.random() < crossover_rate or j == j_rand:
                            trial[j] = mutant[j]
                        else:
                            trial[j] = pop[i][j]
                    
                    trial = clip(trial)
                    trial_f = evaluate(trial)
                    
                    # Selection
                    if trial_f <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(trial_f)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])
                
                if len(new_pop) < lam:
                    break
                
                pop = new_pop
                pop_f = new_pop_f
                
                # Adaptive sigma: shrink if no improvement
                if pop_f[0] >= pop_f[0] * 1.0:  # no improvement check
                    sigma_scale *= 0.95
                
                # Check convergence
                if np.std(pop_f[:lam//2]) < 1e-10:
                    break
            
            # Update best from population
            for i in range(lam):
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = pop[i].copy()

        # Run multi-scale DE
        remaining = budget - calls
        if remaining > 20:
            # Large population for global exploration
            run_de(best_x, remaining // 3, 1.5, 0.7)
            
            if calls < budget and time.time() < deadline:
                # Medium population for basin identification
                run_de(best_x, remaining // 3, 0.5, 0.5)
            
            if calls < budget and time.time() < deadline:
                # Small population for fine-tuning
                run_de(best_x, budget - calls, 0.1, 0.3)

        # Phase 3: Local polish with Nelder-Mead
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining // 2, 'maxfev': remaining,
                             'xatol': 1e-12, 'fatol': 1e-12}
                )
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B polish if budget remains
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    lambda x: evaluate(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': remaining // 2, 'maxfun': remaining,
                             'ftol': 1e-15, 'gtol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f
