"""Katsuura (f23) minimizer using Multi-Start CMA-ES with adaptive restarts.

Strategy:
  1. Initial coarse sampling to identify promising regions.
  2. Multi-start CMA-ES: run several independent CMA-ES instances from different
     starting points. CMA-ES is robust to multimodality and makes efficient use
     of the call budget.
  3. Adaptive restarts: if a CMA-ES instance stagnates, restart from a new point.
  4. Final polish with tight CMA-ES on the best point found.
"""
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
        time_limit = 325.0  # safety margin before 330s kill
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
        for _ in range(20):
            probes.append(rng.uniform(-5, 5, dim))
        # Gaussian around origin
        for _ in range(10):
            probes.append(np.clip(rng.normal(0, 1.5, dim), -5, 5))

        for p in probes:
            if state['calls'] >= budget - 50 or not time_ok():
                break
            call(p)

        best = state['best_x'].copy() if state['best_x'] is not None else np.zeros(dim)

        # ---------- 2. Multi-Start CMA-ES ----------
        try:
            from cma import CMA
        except ImportError:
            from scipy.optimize import minimize as _minimize
            CMA = None

        if CMA is not None:
            # Run multiple CMA-ES instances with adaptive restarts
            n_starts = 5
            calls_per_start = max(60, (budget - state['calls'] - 30) // n_starts)
            
            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 20:
                    break
                
                # Starting point: best found, or random
                if start_idx == 0:
                    x0 = best.copy()
                else:
                    x0 = rng.uniform(-5, 5, dim)
                
                # CMA-ES parameters
                sigma0 = 1.0 if start_idx == 0 else 2.0
                bounds = [(-5.0, 5.0)] * dim
                
                try:
                    es = CMA(x0, sigma0, {
                        'bounds': bounds,
                        'maxiter': 1000,
                        'seed': start_idx + 1,
                        'verbose': -9,
                        'tolopt': 1e-12,
                        'tolfun': 1e-12,
                        'tolfunint': 1e-12,
                    })
                    
                    # Run CMA-ES with call budget tracking (no duplicate evaluations)
                    prev_best = state['best_f']
                    stagnation_count = 0
                    while not es.stop() and state['calls'] < budget - 10 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 10 or not time_ok():
                                fitnesses.append(state['best_f'])
                                break
                            fitnesses.append(call(sol))
                        es.tell(solutions, fitnesses)
                        
                        # Adaptive restart: check for stagnation
                        if state['best_f'] < prev_best - 1e-10:
                            prev_best = state['best_f']
                            stagnation_count = 0
                        else:
                            stagnation_count += 1
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
            n_starts = 3
            for start_idx in range(n_starts):
                if not time_ok() or state['calls'] >= budget - 30:
                    break
                x0 = best.copy() if start_idx == 0 else rng.uniform(-5, 5, dim)
                try:
                    res = minimize(
                        lambda x: call(x),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
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
                    sigma0 = 0.05
                    bounds = [(-5.0, 5.0)] * dim
                    es = CMA(best.copy(), sigma0, {
                        'bounds': bounds,
                        'maxiter': 1000,
                        'seed': 99,
                        'verbose': -9,
                        'tolopt': 1e-14,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                    })
                    while not es.stop() and state['calls'] < budget - 5 and time_ok():
                        solutions = es.ask()
                        fitnesses = []
                        for sol in solutions:
                            if state['calls'] >= budget - 5 or not time_ok():
                                fitnesses.append(state['best_f'])
                                break
                            fitnesses.append(call(sol))
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
                        options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                                 'adaptive': True},
                    )
                    if np.isfinite(res.fun) and res.fun < state['best_f']:
                        state['best_f'] = float(res.fun)
                        state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
                except Exception:
                    pass

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
