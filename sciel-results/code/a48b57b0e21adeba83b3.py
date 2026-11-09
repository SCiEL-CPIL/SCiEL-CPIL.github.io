"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_es(self, func, x0, sigma0, budget, rng, deadline):
        x = x0.copy()
        fx = func(x)
        sigma = sigma0
        success_window = []
        WINDOW = 10
        for _ in range(budget):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(self.dim)
            fy = func(y)
            success_window.append(1 if fy < fx else 0)
            if len(success_window) > WINDOW:
                success_window.pop(0)
            if fy < fx:
                x, fx = y, fy
            # Adapt sigma based on success rate
            if len(success_window) == WINDOW:
                rate = np.mean(success_window)
                if rate > 0.3:
                    sigma *= 1.25
                elif rate < 0.2:
                    sigma *= 0.8
                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-6, 10.0)
        return x, fx

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin from 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial exploration with multiple restarts
        # Try different starting points and step sizes
        best_x = None
        best_fx = np.inf
        
        # Use a portion of budget for exploration
        n_restarts = 5
        budget_per_restart = (self.budget - 50) // n_restarts  # Reserve 50 for polish
        calls_used = 0
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            # Different starting points
            if r == 0:
                x0 = np.zeros(self.dim)
            elif r == 1:
                x0 = rng.uniform(-4, 4, self.dim)
            elif r == 2:
                x0 = rng.uniform(-3, 3, self.dim)
            elif r == 3:
                x0 = np.array([rng.uniform(-4, 4) for _ in range(self.dim)])
            else:
                x0 = best_x if best_x is not None else rng.uniform(-4, 4, self.dim)
            
            sigma0 = 2.0 if r < 3 else 1.0
            
            remaining_budget = min(budget_per_restart, self.budget - calls_used)
            if remaining_budget <= 0:
                break
            
            x, fx = self._run_es(func, x0, sigma0, remaining_budget, rng, deadline)
            calls_used += remaining_budget
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: Polish with smaller steps around best point
        if best_x is not None and time.time() < deadline:
            remaining = self.budget - calls_used
            if remaining > 0:
                # Use smaller sigma for fine-tuning
                x, fx = self._run_es(func, best_x, 0.5, remaining, rng, deadline)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
