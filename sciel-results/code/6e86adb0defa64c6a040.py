"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        
        # Track success rate for adaptive step size
        window_size = 10
        recent_outcomes = []
        
        def coord_descent(x, fx, budget_left, deadline):
            """Efficient coordinate descent with binary search along each axis."""
            nonlocal calls
            for d in range(self.dim):
                if calls >= budget_left or time.time() > deadline:
                    break
                # Binary search for optimal value along dimension d
                lo, hi = -5.0, 5.0
                for _ in range(8):  # 8 iterations of bisection per dimension
                    if calls >= budget_left or time.time() > deadline:
                        break
                    mid = (lo + hi) / 2.0
                    trial = x.copy()
                    trial[d] = mid
                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls += 1
                    if ft < fx:
                        fx, x = ft, trial.copy()
                        # Move lo toward mid if improving
                        lo = max(lo, mid - 0.1)
                    else:
                        hi = min(hi, mid + 0.1)
            return x, fx
        
        # Phase 1: Global exploration with adaptive (1+1)-ES
        # Use ~70% of budget for global search
        global_budget = int(self.budget * 0.7)
        
        for _ in range(global_budget - 1):
            if time.time() > deadline:
                break
            if calls >= self.budget - 1:
                break
            
            # Adaptive sigma based on success rate (1/5 rule approximation)
            if len(recent_outcomes) >= window_size:
                success_rate = sum(recent_outcomes) / window_size
                if success_rate > 0.3:
                    sigma *= 1.15
                elif success_rate < 0.1:
                    sigma *= 0.85
                else:
                    sigma *= 1.0
                recent_outcomes = []
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                recent_outcomes.append(1)
            else:
                recent_outcomes.append(0)
        
        # Phase 2: Local exploitation with coordinate descent + random perturbations
        # Use remaining budget for fine-tuning
        while calls < self.budget - 1:
            if time.time() > deadline:
                break
            
            # Do a round of coordinate descent
            x, fx = coord_descent(x, fx, self.budget, deadline)
            
            if calls >= self.budget - 1:
                break
            
            # Small random perturbation to escape potential local minima
            if calls < self.budget - 5:
                # Try a small random step
                small_sigma = max(0.01, sigma * 0.1)
                y = np.clip(x + small_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma = max(sigma * 0.9, 0.01)
                else:
                    # If perturbation didn't help, try a slightly larger one occasionally
                    if rng.random() < 0.1:
                        med_sigma = max(0.05, sigma * 0.3)
                        y2 = np.clip(x + med_sigma * rng.standard_normal(self.dim), -5, 5)
                        fy2 = func(y2)
                        calls += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
        
        return x, fx
# EVOLVE-BLOCK-END
