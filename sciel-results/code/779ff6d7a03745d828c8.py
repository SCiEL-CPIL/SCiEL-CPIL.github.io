"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # We'll use a multi-start strategy with CMA-ES-like behavior
        # But given the budget is only 500 calls for dim=5, we need to be efficient.
        
        # Strategy: 
        # 1. Do a quick random search / coarse grid to find a good region
        # 2. Then do a local search (Nelder-Mead or CMA-ES) from the best point
        # 3. If time/budget allows, do another restart from a different region
        
        # Given Griewank-Rosenbrock is multimodal, restarts are key.
        # With 500 calls and dim=5, each local search can use ~100-200 calls.
        
        # Let's implement a simple but effective approach:
        # - Use (1+1)-ES with adaptive step size and restarts
        # - Track best solution globally
        # - Restart if no improvement for a while
        
        def local_search(x0, f0, budget_left, sigma_init=1.0):
            nonlocal best_x, best_f, calls_made
            x = x0.copy()
            fx = f0
            sigma = sigma_init
            no_improve = 0
            max_no_improve = 30
            
            while calls_made < self.budget and no_improve < max_no_improve:
                if time.time() > deadline:
                    break
                    
                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break
            
            return x, fx
        
        # Phase 1: Random initialization with multiple starting points
        # Try a few random points first
        n_initial = 10
        candidates = []
        for i in range(n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            candidates.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort by fitness and do local search from the best few
        candidates.sort(key=lambda t: t[1])
        
        # Phase 2: Local search from top candidates
        n_local = 5
        for i in range(min(n_local, len(candidates))):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x0, f0 = candidates[i]
            # Use decreasing sigma for later starts
            sigma_init = 2.0 * (0.7 ** i)
            local_search(x0, f0, self.budget - calls_made, sigma_init)
        
        # Phase 3: If we still have budget, do a final restart from best point with smaller sigma
        if calls_made < self.budget and best_x is not None and time.time() < deadline:
            # Fine-tuning phase
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(20):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
