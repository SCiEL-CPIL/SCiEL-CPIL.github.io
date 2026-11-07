import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Exploration with Differential Evolution
        # Use a moderate population size and iterations to cover the landscape.
        # popsize=10, maxiter=20 -> ~200 evaluations.
        # This is robust for finding the global peak in 5D.
        
        # We allocate 60% of the budget to global search, 40% to local refinement.
        # DE is stochastic, so we run it first.
        
        try:
            # Use latinhypercube initialization for better spread
            de_result = differential_evolution(
                wrapped_func,
                bounds=[(-5.0, 5.0)] * dim,
                maxiter=30,  # Allow up to 30 generations
                popsize=8,   # 8 * 5 = 40 evaluations per gen? No, popsize is number of individuals.
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                init='latinhypercube',
                polish=False  # We will do our own polishing
            )
            # DE uses approx popsize * maxiter calls. 8 * 30 = 240 calls.
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass

        # Phase 2: Local Refinement with L-BFGS-B
        # Start from the best point found by DE.
        # L-BFGS-B is very efficient for smooth local convergence.
        
        if eval_count < budget - 10 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Secondary Global Check / Random Restarts
        # If we have significant budget left, do a few random restarts of L-BFGS-B
        # to ensure we didn't miss a better peak.
        
        remaining = budget - eval_count
        if remaining > 20 and time.time() < deadline:
            rng = np.random.default_rng(123)
            # Try 3-5 random starts
            n_starts = min(5, remaining // 10)
            for _ in range(n_starts):
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                start_pt = rng.uniform(-5.0, 5.0, dim)
                try:
                    res = minimize(
                        wrapped_func,
                        start_pt,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 20, 'maxls': 10, 'ftol': 1e-14, 'gtol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
