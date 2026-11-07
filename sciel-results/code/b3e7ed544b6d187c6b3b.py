import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Cuckoo Search with Lévy flights for global exploration
        # Use ~70% of budget for global search
        cs_budget = int(budget * 0.7)
        
        # Cuckoo Search parameters
        pop_size = 15
        pa = 0.25  # Probability of replacing a nest
        alpha = 1.5  # Lévy flight exponent
        levy_scale = 1.0
        
        # Initialize population
        population = np.random.uniform(-4, 4, (pop_size, n))
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if calls >= cs_budget or not time_ok():
                break
            fitness[i] = func(population[i])
            calls += 1
        
        best_idx = np.argmin(fitness)
        fbest = fitness[best_idx]
        x_best = population[best_idx].copy()
        
        # Lévy flight generation using Mantegna's method
        def levy_flight(scale):
            beta = alpha
            sigma_u = 1.0
            sigma_v = ((1 + beta) * np.sin(np.pi * beta / 2)) / ((beta + 1) * np.exp(-np.pi * beta / 2))
            sigma_v = np.sqrt(sigma_v)
            u = rng.standard_normal(n) * sigma_u
            v = rng.standard_normal(n) * sigma_v
            step = u / np.abs(v) ** (1 / beta)
            return step * scale

        # Run Cuckoo Search
        gen = 0
        no_improve = 0
        while calls < cs_budget and time_ok() and (deadline - time.time()) > 30:
            # Generate new solutions using Lévy flights
            new_pop = np.zeros((pop_size, n))
            new_fit = np.zeros(pop_size)
            
            for i in range(pop_size):
                if calls >= cs_budget or not time_ok():
                    break
                # Lévy flight around current position
                step = levy_flight(levy_scale)
                candidate = population[i] + step
                candidate = np.clip(candidate, -5, 5)
                
                # With probability pa, randomly select a nest
                if rng.random() < pa:
                    j = rng.integers(0, pop_size)
                    candidate = population[j] + levy_flight(levy_scale * 0.5)
                    candidate = np.clip(candidate, -5, 5)
                
                new_pop[i] = candidate
                new_fit[i] = func(candidate)
                calls += 1
            
            # Update population based on fitness
            for i in range(pop_size):
                if new_fit[i] < fitness[i]:
                    population[i] = new_pop[i]
                    fitness[i] = new_fit[i]
                    if fitness[i] < fbest:
                        fbest = fitness[i]
                        x_best = population[i].copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Adapt scale based on improvement
            if no_improve > 10:
                levy_scale *= 0.7
                no_improve = 0
            elif no_improve < 5:
                levy_scale = min(levy_scale * 1.1, 3.0)
            
            # Track best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < fbest:
                fbest = fitness[best_idx]
                x_best = population[best_idx].copy()
            
            gen += 1

        # Phase 2: Multi-start Nelder-Mead for local refinement
        # Generate diverse starting points for local search
        n_nm_starts = 4
        nm_starts = []
        
        # Include the best point from CMA-ES
        nm_starts.append(x_best.copy())
        
        # Add diverse points
        for i in range(1, n_nm_starts):
            if calls >= budget or not time_ok():
                break
            # Different scales around the best point
            scale = 0.5 + i * 0.5
            x_try = x_best + scale * rng.standard_normal(n)
            x_try = np.clip(x_try, -5, 5)
            nm_starts.append(x_try)
        
        # Also add some random points
        for i in range(2):
            if calls >= budget or not time_ok():
                break
            x_try = rng.uniform(-4, 4, n)
            nm_starts.append(x_try)
        
        # Run Nelder-Mead from each start
        for start_x in nm_starts:
            if calls >= budget or not time_ok() or (deadline - time.time()) < 15:
                break
            
            try:
                def wrapped_func(x, _calls_ref=[calls]):
                    nonlocal calls, fbest
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < fbest:
                        fbest = fx
                    return fx

                result = scipy.optimize.minimize(
                    wrapped_func, start_x, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 3: L-BFGS-B for fast local convergence on ill-conditioned problems
        if calls < budget and time_ok() and (deadline - time.time()) > 10:
            try:
                def wrapped_func_lbfgs(x, _calls_ref=[calls]):
                    nonlocal calls, fbest
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < fbest:
                        fbest = fx
                    return fx

                result = scipy.optimize.minimize(
                    wrapped_func_lbfgs, x_best, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 30, 'gtol': 1e-8}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 4: Final fine-tuning
        if calls < budget and time_ok() and (deadline - time.time()) > 5:
            for _ in range(min(10, budget - calls)):
                if not time_ok() or calls >= budget:
                    break
                for scale in [0.01, 0.005, 0.001]:
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x_best + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fbest:
                        fbest = f_try
                        x_best = x_try.copy()

        return x_best, fbest
