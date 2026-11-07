"""Multi-start CMA-ES with Nelder-Mead polish for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start CMA-ES optimizer with Nelder-Mead polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)

        # Phase 1: Coarse random search to identify promising basins.
        # Spend ~30% of budget on random sampling.
        n_random = max(15, budget // 3)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES local refinement on top candidates
        top_k = min(6, len(candidates))

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x0 = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 30:
                break

            # Allocate budget to this CMA-ES run
            cma_budget = max(40, remaining // (top_k - i))
            cma_budget = min(cma_budget, remaining)

            try:
                from cma import CMA
            except ImportError:
                # Fallback: use scipy's Nelder-Mead if CMA-ES not available
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        eval_and_track,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(50, cma_budget),
                            'maxfev': cma_budget,
                            'xatol': 1e-7,
                            'fatol': 1e-9,
                            'adaptive': True,
                        },
                        bounds=[(-5, 5)] * dim,
                    )
                except Exception:
                    pass
                continue

            # CMA-ES parameters
            popsize = max(4, int(4 + 3 * math.log(dim)))
            sigma0 = 1.0  # initial step size

            # Create CMA-ES optimizer
            options = {
                'maxfevals': cma_budget,
                'bounds': [(-5.0, 5.0)] * dim,
                'seed': 42 + i,
                'verbose': -9,
                'tolopt': 1e-8,
                'tolfun': 1e-10,
            }

            try:
                es = CMA(x0, sigma0, options)
                while not es.stop() and calls < budget and time.time() < deadline:
                    solutions = es.ask()
                    for x in solutions:
                        if calls >= budget or time.time() > deadline:
                            break
                        f_val = eval_and_track(x)
                    es.tell(solutions, [eval_and_track(s) for s in solutions])
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish on the current best point
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining >= 15:
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        eval_and_track,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': max(50, remaining // 2),
                            'maxfev': remaining,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        },
                        bounds=[(-5, 5)] * dim,
                    )
                except Exception:
                    pass

        # Phase 4: If budget remains, do adaptive random search around best
        if calls < budget and time.time() < deadline:
            step = 0.5
            min_step = 0.05
            max_step = 2.0
            patience = 0
            max_patience = 30

            while calls < budget and time.time() < deadline:
                # Generate candidate around best point
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)

                if f_val < best_f:
                    patience = 0
                    step = max(min_step, step * 0.9)
                else:
                    patience += 1
                    if patience > max_patience:
                        # Restart with larger step
                        step = min(max_step, step * 1.5)
                        patience = 0
                        # Occasionally jump to a random point
                        if rng.random() < 0.1:
                            x = rng.uniform(-5, 5, size=dim)
                            f_val = eval_and_track(x)
                            if f_val < best_f:
                                step = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
