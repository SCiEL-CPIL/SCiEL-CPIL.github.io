"""Rewritten program: CMA-ES with multi-start, local refinement, and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import cma, minimize


class Optimizer:
    """CMA-ES based optimizer with multi-start and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Multi-start: try several initial points with CMA-ES
        n_starts = min(4, max(1, budget // 150))
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-3, 3, dim))

        for x0 in starts:
            if time.time() >= deadline or call_count >= budget:
                break

            remaining = budget - call_count
            if remaining < 40:
                break

            # CMA-ES
            popsize = max(4, int(4 + 3 * np.log(dim)))
            max_iter = max(2, remaining // (popsize + 1))

            options = {
                'maxiter': max_iter,
                'popsize': popsize,
                'tolfun': 1e-12,
                'tolfunint': 1e-12,
                'verbose': -9,
                'bounds': [(-5.0, 5.0)] * dim,
                'seed': 42,
            }

            try:
                es = cma.CMA(x0, 1.0, options=options)
                while not es.stop():
                    if time.time() >= deadline or call_count >= budget:
                        break
                    solutions = es.ask()
                    fvals = []
                    for sol in solutions:
                        if time.time() >= deadline or call_count >= budget:
                            break
                        fvals.append(wrapped_func(sol))
                    if len(fvals) < len(solutions):
                        # Pad with inf if we ran out of budget mid-population
                        while len(fvals) < len(solutions):
                            fvals.append(np.inf)
                    es.tell(solutions, fvals)
            except Exception:
                pass

            # Local refinement with Nelder-Mead
            if best_x is not None and time.time() < deadline and call_count < budget:
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': min(80, budget - call_count),
                                  'xatol': 1e-8,
                                  'fatol': 1e-10}
                    )
                except Exception:
                    pass

        # Final local search around best
        if best_x is not None and time.time() < deadline and call_count < budget:
            sigma = 0.3
            for _ in range(min(budget - call_count, 100)):
                if time.time() >= deadline:
                    break
                x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fx = wrapped_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-4:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
