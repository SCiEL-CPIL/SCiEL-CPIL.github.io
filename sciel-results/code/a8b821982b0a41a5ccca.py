"""Two-stage optimizer: Sobol global sampling + Nelder-Mead local refinement for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-stage optimizer: deterministic Sobol sampling for global coverage, then Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() < deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or not check_time():
                return None
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sequence sampling for global coverage
        # Use a Sobol sequence to get well-distributed points in [-5, 5]^d
        from scipy.stats import qmc

        # Allocate ~40% of budget to global sampling
        global_budget = int(budget * 0.4)
        n_samples = min(global_budget, 200)  # Cap at 200 Sobol points

        # Generate Sobol sequence
        sampler = qmc.Sobol(d=d, scramble=True, seed=42)
        # Number of points must be a power of 2 for Sobol
        n_points = 1
        while n_points * 2 <= n_samples:
            n_points *= 2

        sobol_samples = sampler.random(n_points)  # in [0, 1]^d
        # Map to [-5, 5]^d
        sobol_samples = sobol_samples * 10.0 - 5.0

        # Evaluate all Sobol samples
        for x in sobol_samples:
            if used_calls >= global_budget or not check_time():
                break
            try_point(x)

        # Phase 2: Nelder-Mead refinement from the best point
        if best_x is not None and used_calls < budget and check_time():
            local_budget = budget - used_calls

            # Run Nelder-Mead from the best Sobol sample
            # Use a custom callback to track function evaluations
            eval_count = [0]

            def wrapped_func(x):
                if eval_count[0] >= local_budget or not check_time():
                    return best_f  # Return best known to stop
                fx = func(x)
                eval_count[0] += 1
                used_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Nelder-Mead with tight tolerances
            try:
                result = minimize(
                    wrapped_func,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_budget,
                        'maxfev': local_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: Additional refinement if budget remains
        if best_x is not None and used_calls < budget and check_time():
            remaining = budget - used_calls

            # Try a few more Nelder-Mead runs from slightly perturbed best_x
            rng = np.random.default_rng(123)
            for _ in range(min(3, remaining // 20)):
                if used_calls >= budget or not check_time():
                    break

                # Small perturbation
                x0 = best_x + rng.normal(0, 0.01, d)
                x0 = np.clip(x0, -5, 5)

                eval_count = [0]

                def wrapped_func2(x):
                    if eval_count[0] >= remaining or not check_time():
                        return best_f
                    fx = func(x)
                    eval_count[0] += 1
                    used_calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                try:
                    result = minimize(
                        wrapped_func2,
                        x0=x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'maxfev': remaining,
                            'xatol': 1e-9,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
