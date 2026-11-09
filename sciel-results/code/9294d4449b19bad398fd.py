"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]

        def safe_func(x):
            if call_count[0] >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        second_best_x = None
        second_best_f = np.inf

        # --- Phase 1: Global Scan ---
        # Focus on finding the basin of attraction quickly.
        # The minimum of Rosenbrock is at 1. With shift in [-4,4], 
        # the minimum is in [-3, 5]. 
        # We sample densely around 1 and sparsely elsewhere.
        
        scan_points = []
        # Dense cluster around 1
        for _ in range(10):
            scan_points.append(rng.normal(1.0, 1.0, size=dim))
        # Dense cluster around 0
        for _ in range(5):
            scan_points.append(rng.normal(0.0, 1.5, size=dim))
        # Broad random
        for _ in range(15):
            scan_points.append(rng.uniform(-4.0, 4.0, size=dim))

        # Evaluate scan points
        for p in scan_points:
            if call_count[0] >= 30:
                break
            if time.time() > deadline:
                break
            f = safe_func(p)
            if f < best_f:
                if best_x is not None:
                    second_best_x = best_x.copy()
                    second_best_f = best_f
                best_f = f
                best_x = p.copy()
            elif f < second_best_f:
                second_best_f = f
                second_best_x = p.copy()

        if best_x is None:
            best_x = np.ones(dim)
            best_f = safe_func(best_x)
            if second_best_x is None:
                second_best_x = np.zeros(dim)
                second_best_f = safe_func(second_best_x)

        # --- Phase 2: Local Optimization with Nelder-Mead ---
        # NM is robust. We give it the majority of the budget.
        
        remaining = budget - call_count[0]
        if remaining > 20:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'maxfun': remaining - 10,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

        # --- Phase 3: Restarts ---
        # If we have budget left, try other candidates.
        
        if call_count[0] < budget - 20 and time.time() < deadline:
            # Try second best from scan
            if second_best_x is not None and second_best_f < best_f * 1.5: # If not too far
                try:
                    res2 = minimize(
                        safe_func,
                        second_best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'maxfun': budget - call_count[0] - 10,
                            'xatol': 1e-8,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res2.fun < best_f:
                        best_x = res2.x
                        best_f = res2.fun
                except Exception:
                    pass

            # Try perturbed best
            if call_count[0] < budget - 20:
                for _ in range(3):
                    if call_count[0] >= budget - 10:
                        break
                    if time.time() > deadline:
                        break
                    perturbation = rng.normal(0, 0.5, size=dim)
                    x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                    try:
                        res3 = minimize(
                            safe_func,
                            x_try,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 100,
                                'maxfun': budget - call_count[0] - 10,
                                'xatol': 1e-8,
                                'fatol': 1e-12,
                                'adaptive': True
                            }
                        )
                        if res3.fun < best_f:
                            best_x = res3.x
                            best_f = res3.fun
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
