"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin within 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution and its value
        best_x = None
        best_f = np.inf

        # Phase 1: Global search using CMA-ES-like strategy
        # We'll implement a simplified CMA-ES with population-based search
        
        # Initial population size and strategy parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of parents
        
        # Initialize center and sigma
        x_center = np.zeros(dim)
        sigma = 1.0
        
        # Weighting for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        B = np.eye(dim)
        D = 1.0
        C = np.eye(dim)
        
        calls_made = 0
        
        # Evaluate initial center
        x_center = np.clip(x_center, -5, 5)
        f_center = func(x_center)
        calls_made += 1
        best_x, best_f = x_center.copy(), f_center
        
        # Generate and evaluate population
        for gen in range(budget // lambda_ + 1):
            # Check deadline
            if time.time() > deadline - 30:  # 30s remaining for refinement
                break
            if calls_made >= budget:
                break
            
            # Generate lambda_ offspring
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                # Generate offspring
                y = x_center + sigma * B @ (D * rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                offspring.append(y)
                offspring_f.append(fy)
                
                if fy < best_f:
                    best_x, best_f = y.copy(), fy
                
                # Check deadline during population eval
                if time.time() > deadline - 30:
                    break
            
            if len(offspring) < 2:
                break
            
            # Sort offspring by fitness
            order = np.argsort(offspring_f)
            offspring_sorted = [offspring[i] for i in order[:mu]]
            offspring_f_sorted = [offspring_f[i] for i in order[:mu]]
            
            # Update center (weighted recombination)
            x_new = np.sum([w * x for w, x in zip(weights, offspring_sorted)], axis=0)
            
            # Update path and covariance
            sigma_old = sigma
            x_center = x_new
            
            # Path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D ** -1) @ (x_new - x_center_old) / sigma_old) if hasattr(self, '_x_center_old') else (1 - c_sigma) * p_sigma
            # Simplified: just update sigma based on success rate
            success = sum(1 for f in offspring_f_sorted if f < f_center)
            if success > 0:
                sigma *= 1.0 + 0.3 * (success / mu - 0.5)
            else:
                sigma *= 0.8
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Update best if center is better
            if f_center > best_f:
                pass  # best already tracked
            
            # Update f_center for next generation
            f_center = offspring_f_sorted[0] if offspring_f_sorted else f_center
            
            # Store for next iteration
            self._x_center_old = x_center_old if hasattr(self, '_x_center_old') else x_new
            x_center_old = x_new
        
        # Phase 2: Local refinement using Nelder-Mead
        remaining_calls = budget - calls_made
        if remaining_calls > 10 and best_x is not None and time.time() < deadline - 10:
            # Try Nelder-Mead refinement from best point
            def wrapped_func(x):
                nonlocal calls_made
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                return fx
            
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_calls - 5, 200),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
