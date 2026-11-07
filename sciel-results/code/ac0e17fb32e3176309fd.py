"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-phase optimizer: diverse sampling + DE + Nelder-Mead for f7 step ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse initial sampling to find good basins quickly
        # Use stratified + random points to cover the space
        n_initial = min(20, budget - 100)
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of strategies: center, corners, random, stratified
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            elif i == 2:
                x = rng.uniform(-4, 4, dim)
            elif i == 3:
                x = np.array([4.0] * dim)
            elif i == 4:
                x = np.array([-4.0] * dim)
            elif i < 8:
                # Stratified: divide each dim into 2 bins
                bin = i % 2
                x = np.array([bin * 8.0 - 4.0 + rng.uniform(0, 4) for _ in range(dim)])
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = _eval(x)
            if fx is None:
                break

        # Phase 2: Differential Evolution (JADE-like) for mid-range optimization
        if calls < budget - 30 and time.time() < deadline - 5:
            # DE parameters
            pop_size = 15  # smaller for tight budget
            max_de_calls = budget - calls - 30  # leave room for NM polish

            # Initialize population around best point with some spread
            pop = []
            pop_f = []
            pop.append(best_x.copy())
            pop_f.append(best_f)
            
            for i in range(pop_size - 1):
                if calls >= budget or time.time() > deadline:
                    break
                # Mix: some near best, some random
                if i < pop_size // 2:
                    x = best_x + rng.standard_normal(dim) * 1.5
                else:
                    x = rng.uniform(-4.5, 4.5, dim)
                x = np.clip(x, -5.0, 5.0)
                fx = _eval(x)
                if fx is None:
                    break
                pop.append(x)
                pop_f.append(fx)
            
            # Sort population
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]
            
            # DE parameters (adaptive)
            F = 0.8  # mutation scale
            CR = 0.7  # crossover rate
            de_generations = 0
            max_generations = max(10, max_de_calls // pop_size)
            
            while calls < budget - 30 and time.time() < deadline - 5 and de_generations < max_generations:
                de_generations += 1
                new_pop = [pop[0].copy()]
                new_pop_f = [pop_f[0]]
                
                # Adaptive F and CR
                F_adapt = max(0.3, F - 0.02 * de_generations)
                CR_adapt = min(0.9, CR + 0.01 * de_generations)
                
                for i in range(1, len(pop)):
                    if calls >= budget - 30 or time.time() > deadline - 5:
                        break
                    
                    # Select three distinct parents (excluding i)
                    candidates = [j for j in range(len(pop)) if j != i]
                    a, b, c = rng.choice(candidates, 3, replace=False)
                    
                    # Mutation: DE/rand/1
                    v = pop[a] + F_adapt * (pop[b] - pop[c])
                    
                    # Crossover: binomial
                    trial = np.empty(dim)
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR_adapt or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][j]
                    
                    trial = np.clip(trial, -5.0, 5.0)
                    fx = _eval(trial)
                    if fx is None:
                        break
                    
                    # Selection
                    if fx <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(fx)
                    else:
                        new_pop.append(pop[i].copy())
                        new_pop_f.append(pop_f[i])
                
                if calls >= budget - 30 or time.time() > deadline - 5:
                    break
                
                # Sort new population
                order = np.argsort(new_pop_f)
                pop = [new_pop[i] for i in order]
                pop_f = [new_pop_f[i] for i in order]
                
                # Early stop if converged
                if pop_f[0] > 0 and np.allclose(pop_f[:5], pop_f[0], atol=1e-6):
                    break

        # Phase 3: Nelder-Mead polish for final convergence
        if calls < budget - 10 and time.time() < deadline - 2:
            # Use scipy's Nelder-Mead if available, else implement simple version
            try:
                from scipy.optimize import minimize
                nm_budget = budget - calls - 5
                nm_options = {
                    'maxiter': nm_budget // 2,
                    'maxfev': nm_budget,
                    'xatol': 1e-6,
                    'fatol': 1e-8
                }
                
                def nm_func(x):
                    nonlocal calls
                    if calls >= budget - 5:
                        return 1e10
                    x = np.clip(x, -5.0, 5.0)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx
                
                result = minimize(nm_func, best_x, method='Nelder-Mead', options=nm_options)
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                # Fallback: simple coordinate descent
                if calls < budget - 10 and time.time() < deadline - 2:
                    x = best_x.copy()
                    f = best_f
                    step = 0.5
                    for _ in range(50):
                        if calls >= budget - 5 or time.time() > deadline - 2:
                            break
                        improved = False
                        for d in range(dim):
                            if calls >= budget - 5:
                                break
                            for sign in [1, -1]:
                                if calls >= budget - 5:
                                    break
                                x_new = x.copy()
                                x_new[d] += sign * step
                                x_new = np.clip(x_new, -5.0, 5.0)
                                fx = _eval(x_new)
                                if fx is None:
                                    break
                                if fx < f - 1e-12:
                                    x = x_new
                                    f = fx
                                    best_x = x.copy()
                                    best_f = f
                                    improved = True
                        if not improved:
                            step *= 0.5
                            if step < 1e-5:
                                break

        return best_x, best_f

# EVOLVE-BLOCK-END
