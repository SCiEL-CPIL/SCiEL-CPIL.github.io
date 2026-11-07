"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Hybrid optimizer: Differential Evolution for global search,
    Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_fx
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial random exploration to get a starting point
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if not time_ok():
                break
            x = rng.uniform(-4, 4, size=dim)
            evaluate(x)

        if global_best_x is None:
            # Should not happen, but just in case
            global_best_x = np.zeros(dim)
            global_best_fx = evaluate(global_best_x)

        # Phase 2: Differential Evolution for global search
        # DE is good at handling the multimodal nature of Büche-Rastrigin
        def de_objective(x):
            if not time_ok():
                return np.inf
            return evaluate(x)

        bounds = [(-5, 5)] * dim

        # Run DE with a moderate population size
        # maxiter is set high but we rely on the budget/time limit
        try:
            # strategy='best1bin' is often good
            # popsize=15 is moderate
            de_result = differential_evolution(
                de_objective,
                bounds,
                strategy='best1bin',
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=False,  # We will do our own polish
                init='latinhypercube'
            )
            if de_result.x is not None:
                fx_de = evaluate(de_result.x)
                if fx_de < global_best_fx:
                    global_best_fx = fx_de
                    global_best_x = de_result.x.copy()
        except Exception:
            pass

        # Phase 3: Local refinement with Nelder-Mead
        # Büche-Rastrigin has smooth basins, so local search can refine well
        if time_ok() and global_best_x is not None:
            from scipy.optimize import minimize

            def nm_objective(x):
                if not time_ok():
                    return np.inf
                return evaluate(x)

            # Try Nelder-Mead from the current best
            try:
                nm_result = minimize(
                    nm_objective,
                    global_best_x,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if nm_result.x is not None:
                    fx_nm = evaluate(nm_result.x)
                    if fx_nm < global_best_fx:
                        global_best_fx = fx_nm
                        global_best_x = nm_result.x.copy()
            except Exception:
                pass

        # Phase 4: Coordinate descent for final precision
        if time_ok() and global_best_x is not None:
            x = global_best_x.copy()
            fx = global_best_fx
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

            for step in step_sizes:
                if not time_ok():
                    break
                improved = True
                while improved and time_ok():
                    improved = False
                    for j in range(dim):
                        if not time_ok():
                            break
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[j] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            fx_trial = evaluate(x_trial)
                            if fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                improved = True
                                break

            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
