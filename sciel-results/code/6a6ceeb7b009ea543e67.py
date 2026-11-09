"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with CMA-ES-like covariance adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Multi-start: use several initial points to avoid poor initialization
        # Split budget among starts, with more budget for later starts if they do well
        n_starts = 5
        # Allocate budget: first start gets 40%, remaining split among others
        budgets_per_start = [int(budget * 0.4)] + [int(budget * 0.6 / (n_starts - 1))] * (n_starts - 1)
        # Ensure sum doesn't exceed budget
        total_alloc = sum(budgets_per_start)
        if total_alloc > budget:
            budgets_per_start[-1] -= (total_alloc - budget)
        
        best_x_global = None
        best_f_global = np.inf

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            
            # Generate initial point for this start
            if start_idx == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Diversify: use different random seeds or strategies
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            best_x = x.copy()
            best_f = fx
            calls_in_start = 1
            
            # Update global best
            if best_f < best_f_global:
                best_f_global = best_f
                best_x_global = best_x.copy()
            
            # Step size and covariance
            sigma = 1.0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Adaptation parameters
            c_1 = 2 / ((dim + 1.3) ** 2 + dim)
            
            # Run (1+1)-ES for this start
            max_calls_this_start = budgets_per_start[start_idx]
            
            for _ in range(max_calls_this_start - 1):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * (C_sqrt @ rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_in_start += 1
                
                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # Update global best
                if best_f < best_f_global:
                    best_f_global = best_f
                    best_x_global = best_x.copy()
                
                # Adaptation
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    
                    # Update sigma using 1/5 rule approximation
                    sigma *= 1.05
                    sigma = max(sigma, 1e-6)
                    sigma = min(sigma, 10.0)
                    
                    # Update covariance (simplified for (1+1))
                    d = (y - x) / sigma
                    C = (1 - c_1) * C + c_1 * (np.outer(d, d) - C)
                    # Ensure positive semi-definite
                    C = (C + C.T) / 2
                    # Periodically recompute square root
                    if calls_in_start % 10 == 0:
                        try:
                            eigvals, eigvecs = np.linalg.eigh(C)
                            eigvals = np.clip(eigvals, 1e-10, None)
                            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                        except:
                            C_sqrt = np.eye(dim)
                            C = np.eye(dim)
                else:
                    # Failure
                    sigma *= 0.95
                    sigma = max(sigma, 1e-6)
                    sigma = min(sigma, 10.0)
                
                # If we're very close to convergence, consider restarting
                if sigma < 0.01 and calls_in_start > 50:
                    break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
