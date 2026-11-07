"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: CMA-ES for global search exploiting valley structure.
    - Phase 3: Multi-start L-BFGS-B local refinement from top candidates.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        n_initial = min(32, self.budget // 8)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 50:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        evaluations.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES for global search exploiting valley structure
        # Use the best initial point as the starting mean
        x_mean = evaluations[0][1].copy()
        sigma = 1.0  # Initial standard deviation
        
        # CMA-ES parameters
        pop_size = max(4, self.dim + 2)  # Population size
        max_cma_iter = 50  # Maximum iterations
        
        # Simple CMA-ES implementation
        def cma_es_step():
            nonlocal x_mean, sigma, calls_used, best_f, best_x
            if calls_used >= self.budget - 30:
                return False
            if time.time() > deadline - 10:
                return False
            
            # Generate population
            population = []
            for i in range(pop_size):
                if calls_used >= self.budget - 30:
                    return False
                if time.time() > deadline - 10:
                    return False
                
                # Sample from Gaussian
                z = rng.standard_normal(self.dim)
                x = x_mean + sigma * z
                x = np.clip(x, -5, 5)
                
                fx = safe_call(x)
                population.append((fx, x))
            
            # Sort by fitness
            population.sort(key=lambda t: t[0])
            
            # Update mean with weighted average of best half
            n_best = pop_size // 2
            weights = np.arange(n_best, 0, -1)
            weights = weights / weights.sum()
            
            new_mean = np.zeros(self.dim)
            for i in range(n_best):
                new_mean += weights[i] * population[i][1]
            
            # Update sigma
            old_sigma = sigma
            sigma = 0.9 * sigma + 0.1 * np.std([p[1] for p in population[:n_best]], axis=0)
            sigma = np.clip(sigma, 0.01, 5.0)
            
            x_mean = new_mean
            return True
        
        # Run CMA-ES for a limited number of iterations
        for _ in range(max_cma_iter):
            if not cma_es_step():
                break
            
            # Update evaluations list with new points
            # (This is simplified; in practice you'd track all evaluations)
            if calls_used >= self.budget - 30:
                break

        # Phase 3: Local refinement from the best few points
        remaining_budget = self.budget - calls_used

        # Select top candidates for local refinement
        n_restarts = min(5, len(evaluations))
        max_iter_per_restart = max(10, remaining_budget // (n_restarts + 1))

        bounds = [(-5, 5)] * self.dim

        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
            if calls_used >= self.budget - 5:
                break

            fx0, x0 = evaluations[i]

            def lbfgs_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    lbfgs_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter_per_restart,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                    }
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try a few more random restarts with local search
        if calls_used < self.budget - 20 and time.time() < deadline - 30:
            possible_restarts = (self.budget - calls_used) // 30
            for _ in range(possible_restarts):
                if time.time() > deadline - 10:
                    break
                if calls_used >= self.budget - 10:
                    break

                x0 = rng.uniform(-5, 5, self.dim)
                fx0 = safe_call(x0)

                if calls_used < self.budget - 5:
                    def lbfgs_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 20,
                                'gtol': 1e-8,
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
