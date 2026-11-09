"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _single_es(self, func, rng, x0, sigma0, budget):
        x = x0.copy()
        fx = func(x)
        sigma = sigma0
        for _ in range(budget):
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, 4.0)
            else:
                sigma = max(sigma * 0.8, 0.01)
        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Use multi-restart strategy
        # First: try a few diverse starting points with limited budget each
        # Then: continue with the best one for remaining budget
        
        # Strategy: split budget among restarts, but keep track of best
        # For Rastrigin, global min is at origin (before rotation/shift)
        # After rotation/shift, it's somewhere in [-4,4]^5
        
        # Try starting from origin first (likely near global min after transform)
        # Then try random points
        
        best_x = None
        best_fx = float('inf')
        
        # Phase 1: Try origin and a few random points with small budgets
        candidates = [np.zeros(self.dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, self.dim))
        
        # Give each candidate a portion of the budget
        # But we need to be time-aware
        n_candidates = len(candidates)
        # Reserve some budget for the final best candidate
        # Let's do: 30 calls per candidate in phase 1, then rest for best
        
        phase1_budget = 20
        for x0 in candidates:
            if time.time() > deadline:
                break
            x, fx = self._single_es(func, rng, x0, 1.5, phase1_budget)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: Continue with the best candidate for remaining budget
        if best_x is not None:
            # Calculate remaining budget accurately
            # Each _single_es call uses 1 (initial) + budget (loop) calls
            # Phase 1 used: n_candidates * (1 + phase1_budget)
            calls_used_phase1 = n_candidates * (1 + phase1_budget)
            
            # We have self.budget total.
            # Phase 2 will use 1 (initial) + remaining_loop_budget (loop)
            # Total = calls_used_phase1 + 1 + remaining_loop_budget <= self.budget
            remaining_loop_budget = self.budget - calls_used_phase1 - 1
            
            if remaining_loop_budget > 0:
                # Continue from best point
                x, fx = self._single_es(func, rng, best_x, 1.0, remaining_loop_budget)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
