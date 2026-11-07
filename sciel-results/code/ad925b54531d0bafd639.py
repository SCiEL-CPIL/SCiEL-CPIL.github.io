"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with 1/5 success rule and final local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        budget = self.budget
        dim = self.dim
        
        # Reserve calls for final refinement
        reserve = max(30, budget // 8)  # ~62 calls reserved
        
        # Portfolio structure: multiple independent (1+1)-ES runs
        n_runs = 4
        calls_per_run = (budget - reserve) // n_runs
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Diverse initial sigmas to cover different scales
        initial_sigmas = [3.0, 2.0, 1.5, 1.0]
        
        for run_idx in range(n_runs):
            if time.time() >= deadline - 20:
                break
            if total_calls >= budget - reserve:
                break
            
            # Each run starts from a different random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            sigma = initial_sigmas[run_idx]
            
            # (1+1)-ES with 1/5 success rule for this run
            # Simple adaptation: multiply by 1.5 on success, 0.87 on failure
            for _ in range(calls_per_run - 1):
                if time.time() >= deadline - 20:
                    break
                if total_calls >= budget - reserve:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Keep sigma within reasonable bounds
                sigma = float(np.clip(sigma, 0.005, 10.0))
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Final local refinement around the best point found
        if best_x is not None and total_calls < budget - 5 and time.time() < deadline - 10:
            x = best_x.copy()
            fx = best_f
            # Start refinement with a small sigma to exploit the best region
            sigma = 0.1
            remaining_calls = budget - total_calls - 5
            
            for _ in range(remaining_calls):
                if time.time() >= deadline - 5:
                    break
                if total_calls >= budget - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                
                # Keep sigma within bounds for refinement
                sigma = float(np.clip(sigma, 0.001, 1.0))
        
        if best_x is None:
            # Fallback: should not happen, but be safe
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
