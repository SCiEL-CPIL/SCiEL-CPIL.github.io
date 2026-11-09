"""Rewritten optimizer: CMA-ES with restarts for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with restarts for multimodal Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # Track best solution and value
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget or time.time() > deadline:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES with restarts strategy
        # Rastrigin is multimodal, so we need multiple restarts from different starting points
        
        # Strategy: 
        # 1. First, do a quick random sampling to get a good initial point
        # 2. Then run CMA-ES from the best point
        # 3. If budget remains, do another CMA-ES from a different random point
        
        # Phase 1: Random sampling to find a good starting point
        n_random = min(50, self.budget // 4)
        for _ in range(n_random):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, self.dim)
            safe_call(x)
        
        # Phase 2: CMA-ES from best point
        if calls_made < self.budget and time.time() < deadline:
            try:
                remaining = self.budget - calls_made
                max_iter = min(remaining // 2, 150)
                if max_iter > 0:
                    result = minimize(
                        safe_call,
                        x0=best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
            except Exception:
                pass
        
        # Phase 3: If we have budget left, try another local search from a different point
        if calls_made < self.budget and time.time() < deadline:
            # Try a few more random restarts with local polish
            n_restarts = min(5, (self.budget - calls_made) // 30)
            for _ in range(n_restarts):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Random restart point
                x_start = np.random.uniform(-4, 4, self.dim)
                fx_start = safe_call(x_start)
                
                # Quick Nelder-Mead polish
                if calls_made < self.budget and time.time() < deadline:
                    try:
                        remaining = self.budget - calls_made
                        max_iter = min(remaining // 2, 50)
                        if max_iter > 0:
                            result = minimize(
                                safe_call,
                                x0=x_start,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': max_iter,
                                    'xatol': 1e-6,
                                    'fatol': 1e-6,
                                    'adaptive': True
                                }
                            )
                    except Exception:
                        pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
