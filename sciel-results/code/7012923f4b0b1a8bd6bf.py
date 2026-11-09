"""Optimizer for BBOB f23 (Katsuura) with multi-restart (1+1)-ES and time guard."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart (1+1)-ES with time guard and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        # Phase 1: Diverse initial sampling to find good starting points
        # Sample 20 points: mix of uniform, Gaussian near origin, and random
        n_samples = 20
        candidates = []
        
        # Uniform samples
        for _ in range(8):
            candidates.append(rng.uniform(-5, 5, dim))
        
        # Gaussian samples centered at origin
        for _ in range(8):
            candidates.append(np.clip(rng.normal(0, 2.0, dim), -5, 5))
        
        # Some near-origin samples
        for _ in range(4):
            candidates.append(np.clip(rng.normal(0, 0.5, dim), -5, 5))
        
        # Evaluate all candidates
        for x in candidates:
            if not time_ok() or calls_made >= budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
        
        # Phase 2: Local search from the best candidate using Nelder-Mead
        if best_x is not None and time_ok() and calls_made < budget:
            # Use Nelder-Mead for local refinement
            def wrapped_func(x):
                nonlocal calls_made
                if not time_ok() or calls_made >= budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                return fx
            
            # Run Nelder-Mead with limited function evaluations
            nm_options = {
                'maxiter': 100,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', options=nm_options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: If budget remains, do a fine-grained (1+1)-ES polish
        if best_x is not None and time_ok() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            success_count = 0
            total_count = 0
            max_iter = budget - calls_made
            
            for _ in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    success_count += 1
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                
                total_count += 1
                if total_count % 10 == 0:
                    rate = success_count / total_count
                    if rate > 0.2:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.8
                    success_count = 0
                    total_count = 0
                
                sigma = np.clip(sigma, 0.001, 2.0)
        
        return best_x, best_f
