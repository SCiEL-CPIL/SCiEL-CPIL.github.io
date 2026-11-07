"""Multi-start Differential Evolution with adaptive (1+1)-ES polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Multi-start DE with adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0

        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except:
                return np.inf

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Multi-run Differential Evolution for global exploration
        # Use multiple runs with decreasing budgets to find good basins
        runs = [250, 150, 60]

        for i, run_budget in enumerate(runs):
            if n_calls[0] >= budget or time.time() > deadline:
                break

            stop_flag[0] = False
            start_calls = n_calls[0]

            try:
                res = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    tol=1e-10,
                    seed=42 + i,
                    init='latinhypercube',
                    callback=lambda xk, convergence: True if (
                        stop_flag[0] or
                        n_calls[0] - start_calls >= run_budget or
                        time.time() > deadline
                    ) else False
                )

                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Adaptive (1+1)-ES with two-stage polish
        remaining = budget - n_calls[0]
        if remaining > 5 and time.time() < deadline:
            # Initialize step size
            sigma = 0.5
            x = best_x.copy()
            f_x = best_f
            stagnation = 0
            es_calls = 0

            # Two-stage polish: medium then fine
            # First 70% of ES budget: medium sigma
            # Last 30%: fine sigma with progressive reduction
            es_budget = remaining
            stage1_calls = int(es_budget * 0.7)

            while n_calls[0] < budget and time.time() < deadline and not stop_flag[0]:
                # Determine current stage
                in_stage2 = es_calls >= stage1_calls

                # Generate one offspring
                rng = np.random.default_rng(hash((n_calls[0], es_calls)) % (2**32))
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)

                # Evaluate
                f_trial = safe_func(x_trial)
                es_calls += 1

                if not np.isfinite(f_trial):
                    break

                # Update best if improved
                if f_trial < f_x:
                    x = x_trial.copy()
                    f_x = f_trial
                    best_x = x.copy()
                    best_f = f_x
                    stagnation = 0
                    # Increase sigma on success (faster adaptation)
                    sigma *= 1.3
                else:
                    stagnation += 1
                    # Decrease sigma on failure
                    sigma *= 0.75

                    # Restart sigma if stagnated
                    if stagnation > 50:
                        sigma = max(sigma, 0.3)
                        stagnation = 0

                # Two-stage sigma control
                if in_stage2:
                    # Force smaller sigma in the last 30%
                    progress_in_stage2 = (es_calls - stage1_calls) / max(1, es_budget - stage1_calls)
                    if progress_in_stage2 > 0.5:
                        # Fine polish: progressively smaller sigma
                        sigma = min(sigma, 0.05)
                    else:
                        sigma = min(sigma, 0.1)

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-5, 5.0)

        # Phase 3: Nelder-Mead polish if budget remains
        # This exploits local smoothness for faster convergence in later stages
        remaining = budget - n_calls[0]
        if remaining > 10 and time.time() < deadline:
            # Nelder-Mead uses dim+1 calls per iteration roughly
            maxiter = max(1, remaining // (dim + 1))

            nm_calls = [0]
            def nm_func(x):
                if nm_calls[0] >= remaining:
                    return np.inf
                nm_calls[0] += 1
                return safe_func(x)

            try:
                res_nm = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < best_f and np.isfinite(res_nm.fun):
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
