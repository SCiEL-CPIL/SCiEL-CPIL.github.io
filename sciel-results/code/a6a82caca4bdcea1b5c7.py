"""Katsuura (f23) minimizer using Multi-Start CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Katsuura (f23) in 5 dimensions using Multi-Start CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 310.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        state = {'calls': 0, 'best_x': None, 'best_f': float('inf')}
        rng = np.random.default_rng(12345)

        def time_ok():
            return time.time() - start_time < time_limit

        def call(x):
            """Evaluate one point, update best, respect budget and time."""
            if state['calls'] >= budget or not time_ok():
                return state['best_f']
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return state['best_f']
            if not np.isfinite(v):
                v = state['best_f']
                return v
            state['calls'] += 1
            if v < state['best_f']:
                state['best_f'] = v
                state['best_x'] = x.copy()
            return v

        # ---------- 1. Initial coarse sampling ----------
        probes = [np.zeros(dim)]
        # Uniform random points
        for _ in range(35):
            probes.append(rng.uniform(-5, 5, dim))
        # Gaussian around origin - denser near center
        for _ in range(20):
            probes.append(np.clip(rng.normal(0, 0.8, dim), -5, 5))
        # Additional points at moderate distance
        for _ in range(10):
            probes.append(np.clip(rng.normal(0, 2.0, dim), -5, 5))

        for p in probes:
            if state['calls'] >= budget - 150 or not time_ok():
                break
            call(p)

        best = state['best_x'].copy() if state['best_x'] is not None else np.zeros(dim)

        # ---------- 2. Multi-Start CMA-ES ----------
        try:
            from cma import CMA
        except ImportError:
            CMA = None

        if CMA is not None:
            # Run multiple CMA-ES instances
            n_starts = 5
            remaining_budget = budget - state['calls'] - 50
            start_budgets = []
            if remaining_budget > 0:
                # First start gets 35%, rest split among remaining
                first_budget = int(remaining_budget * 0.35)
                rest_budget = remaining_budget - first_budget
                per_start = max(25, rest_budget // (n_starts - 1))
                start_budgets.append(first_budget)
                for _ in range(n_starts - 1):
                    start_budgets.append(per_start)
            else:
                start_budgets = [0] * n_starts

            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 15:
                    break

                # Starting point: best found for first, then random
                if start_idx == 0:
                    x0 = best.copy()
                    sigma0 = 0.3
                else:
                    x0 = rng.uniform(-4, 4, dim)
                    sigma0 = 1.5

                # CMA-ES parameters
                bounds = [(-5.0, 5.0)] * dim
                max_calls_this_start = start_budgets[start_idx] if start_idx < len(start_budgets) else 50

                try:
                    es = CMA(x0, sigma0, {
                        'bounds': bounds,
                        'maxiter': 3000,
                        'seed': start_idx + 200,
                        'verbose': -9,
                        'tolopt': 1e-14,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                        'popsize': 10,
                        'cc': 0.2,
                        'cs': 0.1,
                    })

                    calls_this_start = 0
                    stagnation_count = 0
                    prev_best = state['best_f']

                    while not es.stop() and state['calls'] < budget - 15 and time_ok() and calls_this_start < max_calls_this_start:
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 15 or not time_ok() or calls_this_start >= max_calls_this_start:
                                break
                            v = call(sol)
                            fitnesses.append(v)
                            calls_this_start += 1

                        if len(fitnesses) < len(solutions):
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))

                        es.tell(solutions, fitnesses)

                        # Check for stagnation
                        if state['best_f'] < prev_best - 1e-10:
                            stagnation_count = 0
                            prev_best = state['best_f']
                        else:
                            stagnation_count += 1

                        # If stagnating for too long, break early
                        if stagnation_count > 30:
                            break

                    # Update best from CMA-ES result
                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                            best = xb.copy()

                except Exception:
                    pass

        else:
            # Fallback: use scipy Nelder-Mead if CMA-ES not available
            from scipy.optimize import minimize
            n_starts = 4
            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 40:
                    break
                x0 = best.copy() if start_idx == 0 else rng.uniform(-4, 4, dim)
                try:
                    res = minimize(
                        lambda x: call(x),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                        best = state['best_x'].copy()
                except Exception:
                    pass

        # ---------- 3. Final polish with tight CMA-ES ----------
        if time_ok() and state['calls'] < budget - 10:
            if CMA is not None:
                try:
                    sigma0 = 0.02
                    bounds = [(-5.0, 5.0)] * dim
                    es = CMA(best.copy(), sigma0, {
                        'bounds': bounds,
                        'maxiter': 2000,
                        'seed': 888,
                        'verbose': -9,
                        'tolopt': 1e-16,
                        'tolfun': 1e-16,
                        'tolfunint': 1e-16,
                        'popsize': 8,
                        'cc': 0.3,
                        'cs': 0.15,
                    })

                    while not es.stop() and state['calls'] < budget - 5 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 5 or not time_ok():
                                break
                            v = call(sol)
                            fitnesses.append(v)

                        if len(fitnesses) < len(solutions):
                            worst = max(fitnesses) if fitnesses else state['best_f']
                            fitnesses.extend([worst] * (len(solutions) - len(fitnesses)))

                        es.tell(solutions, fitnesses)

                    if es.result.xbest is not None:
                        xb = np.clip(np.asarray(es.result.xbest, dtype=float), -5, 5)
                        fb = call(xb)
                        if fb < state['best_f']:
                            state['best_f'] = fb
                            state['best_x'] = xb.copy()
                except Exception:
                    pass
            else:
                from scipy.optimize import minimize
                try:
                    res = minimize(
                        lambda x: call(x),
                        best.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                except Exception:
                    pass

        # ---------- 4. Local refinement with perturbation ----------
        if time_ok() and state['calls'] < budget - 5:
            # Try small perturbations around the best point
            for _ in range(20):
                if state['calls'] >= budget - 5 or not time_ok():
                    break
                perturb_scale = 0.01
                x_perturb = best + rng.normal(0, perturb_scale, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                v = call(x_perturb)
                if v < state['best_f']:
                    state['best_f'] = v
                    state['best_x'] = x_perturb.copy()
                    best = x_perturb.copy()

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
