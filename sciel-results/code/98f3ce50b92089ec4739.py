import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Quick random sampling for early progress (AOCC)
        n_initial = min(10, budget)
        rng = np.random.default_rng(42)
        for _ in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            wrapper(x)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        # Phase 2: Differential Evolution for global search
        # DE is robust to the rotated Rosenbrock landscape
        de_budget = int(budget * 0.55)
        remaining_for_de = de_budget - calls_made
        if remaining_for_de > 15:
            # popsize * maxiter ≈ remaining_for_de
            # Use popsize=15 for 5D, adjust maxiter
            popsize = 15
            maxiter_de = max(1, remaining_for_de // popsize)
            
            try:
                differential_evolution(
                    wrapper,
                    bounds=[(-5.0, 5.0)] * dim,
                    popsize=popsize,
                    maxiter=maxiter_de,
                    seed=42,
                    tol=1e-10,
                    init='latinhypercube',
                    polish=False,
                    update_interval=1
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        # Phase 3: L-BFGS-B local refinement
        # Very efficient for smooth Rosenbrock once in the valley
        if calls_made < budget and time.time() < deadline - 10.0:
            try:
                minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 500, 'gtol': 1e-15, 'ftol': 1e-15}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try Nelder-Mead from best point
        if calls_made < budget and time.time() < deadline - 5.0:
            try:
                minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
                )
            except Exception:
                pass

        # Phase 5: Final random perturbations if budget remains
        if calls_made < budget and time.time() < deadline - 1.0:
            step_sizes = [0.5, 0.1, 0.01]
            for ss in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break
                for _ in range(5):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    perturbation = rng.uniform(-ss, ss, size=dim)
                    x_new = best_x + perturbation
                    x_new = np.clip(x_new, -5.0, 5.0)
                    wrapper(x_new)

        return best_x, best_f
