import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        all_points = []  # (f, x) for tracking

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration - 25 random points
        n_explore = 25
        for _ in range(n_explore):
            x = rng.uniform(-5, 5, dim)
            eval_func(x)

        # Sort by fitness to find top starting points
        all_points.sort(key=lambda p: p[0])
        
        # Get top 3 distinct starting points (ensuring diversity)
        start_points = []
        for f, x in all_points:
            if len(start_points) >= 3:
                break
            # Ensure minimum distance from existing starts
            if all(np.linalg.norm(x - sp) > 1.5 for sp in start_points):
                start_points.append(x)
        
        # If we don't have 3 diverse points, just take the top ones
        while len(start_points) < 3 and len(all_points) > len(start_points):
            f, x = all_points[len(start_points)]
            if x not in start_points:
                start_points.append(x)

        # Phase 2: CMA-ES from top starting points
        # Budget remaining: 500 - 25 = 475
        # Split among 3 starts: ~150 each, but with early stopping
        
        cma_configs = [
            {'sigma': 1.0, 'popsize': 6, 'maxiter': 25},
            {'sigma': 0.8, 'popsize': 6, 'maxiter': 20},
            {'sigma': 1.2, 'popsize': 6, 'maxiter': 15},
        ]
        
        for i, sp in enumerate(start_points):
            if time.time() > deadline - 10 or calls_made >= budget - 10:
                break
            
            config = cma_configs[i] if i < len(cma_configs) else cma_configs[-1]
            
            try:
                minimize(
                    eval_func,
                    sp,
                    method='CMA',
                    options={
                        'maxiter': config['maxiter'],
                        'tol': 1e-8,
                        'sigma': config['sigma'],
                        'popsize': config['popsize'],
                    }
                )
            except Exception:
                break

        # Phase 3: If budget remains, do a final refinement from best point
        remaining = budget - calls_made
        if remaining > 30 and time.time() < deadline - 5:
            try:
                minimize(
                    eval_func,
                    best_x,
                    method='CMA',
                    options={
                        'maxiter': min(15, remaining // 6),
                        'tol': 1e-10,
                        'sigma': 0.3,
                        'popsize': 5,
                    }
                )
            except Exception:
                pass

        return best_x, best_f
