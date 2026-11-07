"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        total_budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline

        # Phase 1: Multi-start exploration
        # We will try a few different starting points to avoid getting stuck
        # in a poor local basin early on.
        # We allocate a small fraction of the budget to initial exploration.
        # Let's say we try 3 different starts, each with a small number of steps,
        # or we just do a quick random search to find a good starting point.
        
        # A simpler approach: just start at center and let (1+1)-ES do its job,
        # but maybe initialize sigma better.
        # The insight says "Multi-start initialization avoids wasting budget climbing from a poor random start".
        # Let's do a few quick evaluations to pick a better start.
        
        # Strategy: Evaluate a few random points and the center, pick the best one as start.
        # This costs a few calls but ensures we start closer to the optimum.
        
        candidates = []
        # Center
        candidates.append(np.zeros(n))
        # A few random points
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, n))
        
        # Evaluate them
        best_start_x = None
        best_start_f = np.inf
        for c in candidates:
            c = np.clip(c, -5, 5)
            if check_deadline():
                break
            fc = func(c)
            calls_made += 1
            if fc < best_start_f:
                best_start_f = fc
                best_start_x = c.copy()
            if calls_made >= total_budget:
                break
        
        if best_start_x is None:
            # Fallback if something went wrong (shouldn't happen)
            best_start_x = np.zeros(n)
            best_start_f = func(best_start_x)
            calls_made += 1

        x = best_start_x
        fx = best_start_f
        best_x = x.copy()
        best_f = fx

        # Phase 2: (1+1)-ES refinement
        # Adaptive sigma: start with a moderate step size
        sigma = 1.0
        
        # We have remaining budget
        while calls_made < total_budget and not check_deadline():
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                # Increase step size on success
                sigma *= 1.2
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Decrease step size on failure
                sigma *= 0.8
                # If sigma becomes very small, we might be stuck.
                # We could restart or just continue with tiny steps.
                # For Weierstrass, the function is noisy but continuous.
                # Let's keep going.
                
            # Safety check for sigma
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10:
                sigma = 10.0

        return best_x, best_f
# EVOLVE-BLOCK-END
