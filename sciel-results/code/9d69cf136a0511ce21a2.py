"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based ES with adaptive step size and restarts for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def run_es(x0, sigma0, max_calls, deadline, remaining_calls):
            """(1+lambda)-ES with adaptive step size."""
            x = x0.copy()
            sigma = sigma0
            best_x = x.copy()
            best_f = func(x)
            calls = 1
            remaining_calls -= 1
            
            # Success rate tracking for adaptive sigma
            success_rate = 0.5
            target_success = 0.25
            lam = 8  # population size
            
            for it in range(max_calls // lam + 1):
                if time.time() > deadline or remaining_calls <= 0:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if remaining_calls <= 0 or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    remaining_calls -= 1
                    offspring.append((fy, y))
                
                if len(offspring) == 0:
                    break
                    
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                
                # Select best offspring as new center
                x_new = offspring[0][1].copy()
                
                # Adaptive step size: increase if success rate > target, decrease otherwise
                # Count successes (offspring better than parent)
                successes = sum(1 for fy, _ in offspring if fy < best_f + 1e-10)
                success_rate = 0.7 * success_rate + 0.3 * (successes / lam)
                
                if success_rate > target_success:
                    sigma *= 1.2
                elif success_rate < target_success * 0.5:
                    sigma *= 0.8
                else:
                    sigma *= 1.05
                
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)
                
                # Move to best offspring
                x = x_new
                
            return best_x, best_f, calls, remaining_calls

        best_x = None
        best_f = np.inf
        remaining_calls = budget

        # Phase 1: Diverse exploration with multiple restarts
        # Use Sobol-like or grid-based diverse starting points
        start_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            rng.uniform(-3, 3, n),
            rng.uniform(-4, 4, n),
            rng.uniform(-3, 3, n),
        ]
        
        # Allocate budget: ~60% for exploration, 40% for exploitation
        explore_budget = int(budget * 0.6)
        calls_per_restart = explore_budget // len(start_points)
        
        for i, x0 in enumerate(start_points):
            if time.time() > deadline or remaining_calls <= 0:
                break
            
            sigma0 = 2.5 if i < 3 else 1.5
            max_calls = min(calls_per_restart, remaining_calls)
            
            x_best, f_best, calls_used, rem = run_es(x0, sigma0, max_calls, deadline, remaining_calls)
            remaining_calls = rem
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best
        
        # Phase 2: Exploitation around best point with smaller sigma
        if best_x is not None and time.time() < deadline and remaining_calls > 0:
            exploit_calls = min(remaining_calls, budget // 3)
            x_best, f_best, calls_used, rem = run_es(best_x, 0.5, exploit_calls, deadline, remaining_calls)
            remaining_calls = rem
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best
        
        # Phase 3: Fine-tuning with very small sigma
        if best_x is not None and time.time() < deadline and remaining_calls > 0:
            fine_calls = min(remaining_calls, budget // 10)
            x_best, f_best, calls_used, rem = run_es(best_x, 0.1, fine_calls, deadline, remaining_calls)
            remaining_calls = rem
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        return best_x, best_f
# EVOLVE-BLOCK-END
