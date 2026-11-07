"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based optimizer with multi-start and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # State variables
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        def time_exceeded():
            return time.time() > deadline

        def budget_exhausted():
            return call_count >= self.budget

        # Phase 1: Multi-start Nelder-Mead
        # Use a diverse set of starting points to cover the search space
        # Schwefel's function has many local minima, so multi-start is crucial
        
        # Generate initial points: mix of random, grid-like, and centered
        n_starts = 8
        start_points = []
        
        # Center point
        start_points.append(np.zeros(self.dim))
        
        # Random points
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, self.dim))
        
        # Allocate budget per start
        # Reserve some budget for final refinement
        reserve = 50
        per_start_budget = max(20, (self.budget - reserve) // n_starts)
        
        for sp in start_points:
            if time_exceeded() or budget_exhausted():
                break
            
            calls_this = [0]
            
            def nm_obj(x):
                nonlocal calls_this
                if calls_this[0] >= per_start_budget or time_exceeded() or budget_exhausted():
                    return best_f
                calls_this[0] += 1
                return eval_func(np.clip(x, -5, 5))
            
            # Create initial simplex
            simplex = [sp.copy()]
            for i in range(self.dim):
                point = sp.copy()
                step = 0.1 if abs(sp[i]) > 1e-10 else 0.1
                point[i] += step
                simplex.append(np.clip(point, -5, 5))
            
            try:
                res = minimize(
                    nm_obj,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': per_start_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Phase 2: Local refinement around the best point using Nelder-Mead
        # with a tighter tolerance and smaller initial step
        if best_x is not None and call_count < self.budget and not time_exceeded():
            remaining = self.budget - call_count
            if remaining > 20:
                calls_this = [0]
                
                def nm_refine_obj(x):
                    nonlocal calls_this
                    if calls_this[0] >= remaining or time_exceeded() or budget_exhausted():
                        return best_f
                    calls_this[0] += 1
                    return eval_func(np.clip(x, -5, 5))
                
                # Start from the best point found
                try:
                    res = minimize(
                        nm_refine_obj,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final fine-tuning with small random perturbations
        if best_x is not None and call_count < self.budget and not time_exceeded():
            remaining = self.budget - call_count
            for _ in range(min(remaining, 30)):
                if time_exceeded() or budget_exhausted():
                    break
                # Very small step size for fine tuning
                y = np.clip(best_x + 0.005 * rng.standard_normal(self.dim), -5, 5)
                eval_func(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
