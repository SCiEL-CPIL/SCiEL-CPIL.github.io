import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Multi-scale coarse-to-fine random search
        # Start with large steps to find broad basins, then reduce step size
        # Weierstrass has self-similar structure, so multi-scale helps
        
        scales = [3.0, 1.0, 0.3, 0.1]  # Step sizes for each scale
        calls_per_scale = [8, 15, 30, 50]  # Number of random points per scale
        
        for scale, n_points in zip(scales, calls_per_scale):
            for _ in range(n_points):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                try:
                    fx = wrapper(x)
                except TimeoutError:
                    break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if calls_made >= total_budget or time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: 1+1-ES with adaptive sigma and periodic Nelder-Mead polish
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Smaller initial sigma since we already did coarse search

        nm_interval = 30  # Do NM polish every N ES iterations
        es_count = 0
        nm_maxfev = 60  # Max function evaluations for NM polish

        while calls_made < total_budget and time.time() < deadline:
            # Check if we should do NM polish
            if es_count > 0 and es_count % nm_interval == 0 and (total_budget - calls_made) > nm_maxfev + 5:
                # Nelder-Mead polish on current best
                try:
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={
                                       'maxfev': nm_maxfev,
                                       'xatol': 1e-8,
                                       'fatol': 1e-8,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        x = best_x.copy()
                        fx = best_f
                        # Reset sigma after NM (it found a better point)
                        sigma = 0.3
                except TimeoutError:
                    break
                except Exception:
                    pass

            if time.time() > deadline or calls_made >= total_budget:
                break

            # 1+1-ES step
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            try:
                fy = wrapper(y)
            except TimeoutError:
                break

            if fy < fx:
                # Success: move and expand
                x = y
                fx = fy
                sigma *= 1.05
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure: shrink
                sigma *= 0.90

            # Clamp sigma
            sigma = np.clip(sigma, 0.001, 2.0)
            es_count += 1

        # Phase 3: Final high-precision polish if we have budget left
        if calls_made < total_budget - 20 and time.time() < deadline - 10:
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={
                                   'maxfev': min(80, total_budget - calls_made - 5),
                                   'xatol': 1e-10,
                                   'fatol': 1e-10,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
