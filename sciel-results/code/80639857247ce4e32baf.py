"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Strategy:
    1. Structured exploration to find good basins
    2. Focused (1+1)-ES with adaptive step size from best point
    3. Periodic structure exploitation via offset sampling
    4. Final polish with moderate sigma
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Structured Exploration
        # Sample integer lattice points and random continuous points
        
        n_explore = min(60, budget // 5)
        
        # Evaluate integer lattice points (likely near local minima)
        n_lattice = n_explore // 2
        for _ in range(n_lattice):
            if budget_used >= budget or not time_ok():
                break
            pt = rng.integers(-4, 5, size=dim).astype(float)
            pt = np.clip(pt, -5, 5)
            f_val = func(pt)
            budget_used += 1
            update_best(pt, f_val)
        
        # Evaluate random continuous points
        n_random = n_explore - n_lattice
        for _ in range(n_random):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            f_val = func(pt)
            budget_used += 1
            update_best(pt, f_val)

        # Phase 2: Focused (1+1)-ES from best point
        # Run adaptive local search focused on the best point found
        
        if budget_used < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            min_sigma = 0.001
            max_sigma = 5.0
            window_size = 10
            success_window = []
            
            while budget_used < budget and time_ok():
                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                # Update best and track success
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    update_best(x, fx)
                else:
                    success_window.append(0)
                
                # Adapt sigma based on 1/5th success rule
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.08
                    elif success_rate < 0.1:
                        sigma *= 0.92
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
                
                # If converged, break to try periodic offsets
                if sigma < 0.01 and len(success_window) >= window_size:
                    if np.mean(success_window[-window_size:]) < 0.05:
                        break

        # Phase 3: Periodic structure exploitation
        # Büche-Rastrigin has periodic minima; sample near integer offsets from best
        
        if budget_used < budget and time_ok():
            n_offsets = min(20, (budget - budget_used) // 3)
            for _ in range(n_offsets):
                if budget_used >= budget or not time_ok():
                    break
                
                # Sample offset near best point (within [-1, 1] per dimension)
                offset = rng.uniform(-1.0, 1.0, dim)
                pt = np.clip(best_x + offset, -5, 5)
                f_val = func(pt)
                budget_used += 1
                update_best(pt, f_val)
                
                # If this is better, do a quick local refinement
                if f_val < best_f:
                    x = pt.copy()
                    fx = f_val
                    sigma = 0.3
                    for _ in range(min(5, budget - budget_used)):
                        if budget_used >= budget or not time_ok():
                            break
                        y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        budget_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            update_best(x, fx)

        # Phase 4: Final polish
        # Use moderate sigma to escape shallow local minima
        
        remaining = budget - budget_used
        if remaining > 5 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2  # Moderate step size
            min_sigma = 1e-4
            max_sigma = 1.0
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    update_best(x, fx)
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.15
                    elif success_rate < 0.1:
                        sigma *= 0.85
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
