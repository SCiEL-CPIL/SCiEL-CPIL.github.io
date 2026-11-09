"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and stall detection for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        # Strategy: multiple restarts with increasing exploration
        # Büche-Rastrigin has many local minima, so restarts help escape them.
        
        # Initial broad exploration with several random starts
        num_initial_starts = 5
        for i in range(num_initial_starts):
            if calls >= self.budget or time.time() > deadline:
                break
            if i == 0:
                # First start: center
                x = np.zeros(self.dim)
            else:
                # Subsequent: random points
                x = rng.uniform(-4, 4, self.dim)
            
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Now do focused local search with restarts
        sigma_min = 0.01
        sigma_max = 5.0
        stall_threshold = 50  # iterations without improvement before restart
        max_restart_attempts = 20
        
        for attempt in range(max_restart_attempts):
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Decide where to restart from
            if attempt == 0:
                x = best_x.copy() if best_x is not None else rng.uniform(-4, 4, self.dim)
                sigma = 1.0
            elif attempt % 3 == 0:
                # Periodically try a fresh random start
                x = rng.uniform(-4, 4, self.dim)
                sigma = 2.0
            else:
                # Perturb the current best
                x = best_x.copy() + rng.normal(0, 1.5, self.dim)
                x = np.clip(x, -5, 5)
                sigma = 0.8
            
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local (1+1)-ES with stall detection
            stall_count = 0
            max_local_steps = 80  # limit steps per local search to allow restarts
            
            for step in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, sigma_max)
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        break
            
            if time.time() > deadline:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
