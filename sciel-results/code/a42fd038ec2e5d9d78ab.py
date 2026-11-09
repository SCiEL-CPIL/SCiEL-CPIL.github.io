"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio-based black-box optimizer for Schwefel-style f20 on [-5,5]^5.
    Uses a deadline guard to stay within wall-clock limits.
    Strategy:
      1. Quick random restarts to find a decent initial point.
      2. CMA-ES-like adaptive step-size with restarts.
      3. If time remains, try L-BFGS-B from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = None
        best_f = np.inf
        
        def call_func(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        n_random = min(50, budget)
        for _ in range(n_random):
            if time_left() < 5:
                break
            x = rng.uniform(-5, 5, dim)
            call_func(x)
        
        if best_x is None:
            # Should not happen, but safeguard
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        # Phase 2: CMA-ES-like optimization
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        
        # Adaptive step size parameters
        sigma_min = 1e-8
        sigma_max = 10.0
        
        # Success rate tracking for 1/5 rule
        successes = 0
        total = 0
        window_size = 10
        
        # CMA-ES with restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time_left() < 10:
                break
            
            # Reset for each restart if needed
            if restart > 0:
                # Try a different starting point
                for _ in range(10):
                    if time_left() < 5:
                        break
                    x_new = rng.uniform(-5, 5, dim)
                    fx_new = func(x_new)
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        best_f = fx
                        best_x = x.copy()
                        break
                # If no improvement, keep current x but reset sigma
                sigma = 1.0
                successes = 0
                total = 0
            
            # Run CMA-ES iterations
            iter_count = 0
            while iter_count < budget - n_random and time_left() > 10:
                iter_count += 1
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                total += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    # Increase step size on success
                    sigma = min(sigma * 1.2, sigma_max)
                    # Update best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.8, sigma_min)
                
                # 1/5 success rule: check every window_size steps
                if total % window_size == 0:
                    success_rate = successes / total
                    if success_rate > 0.4:
                        sigma = min(sigma * 1.5, sigma_max)
                    elif success_rate < 0.1:
                        sigma = max(sigma * 0.5, sigma_min)
                    successes = 0
                    total = 0
                
                # Convergence check: if sigma is very small, break
                if sigma < sigma_min * 10:
                    break
        
        # Phase 3: Local refinement with Nelder-Mead or similar if time allows
        if time_left() > 20 and best_x is not None:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
