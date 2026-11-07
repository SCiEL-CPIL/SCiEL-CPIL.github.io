"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Two-stage hybrid: CMA-ES global search then L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s kill

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (d,):
                x = x.reshape(d)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = best_fx if np.isfinite(best_fx) else 1e10
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES global search
        # CMA-ES is effective for smooth multimodal functions like Discus.
        # We'll implement a simplified CMA-ES to control the budget.
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))  # ~11 for d=5
        sigma = 2.0  # Initial step size
        x_mean = np.zeros(d)
        
        # CMA-ES weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Covariance matrix
        C = np.eye(d)
        C_inv = np.eye(d)
        
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Initial population
        rng = np.random.default_rng(42)
        
        max_feas_cma = 200  # Use 200 calls for CMA-ES
        cma_calls = 0
        
        def cma_eval(x):
            nonlocal cma_calls
            if cma_calls >= max_feas_cma:
                return best_fx
            return safe_func(x)

        # Run CMA-ES
        generation = 0
        while cma_calls < max_feas_cma and time_ok():
            # Generate population
            offspring = []
            for i in range(pop_size):
                # Sample from N(x_mean, sigma^2 * C)
                # Use simple diagonal approximation
                z = rng.standard_normal(d)
                x_off = x_mean + sigma * z
                x_off = np.clip(x_off, -5.0, 5.0)
                offspring.append((cma_eval(x_off), x_off))
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            new_x_mean = np.zeros(d)
            for i in range(mu):
                new_x_mean += weights[i] * offspring[i][1]
            
            # Update sigma
            old_sigma = sigma
            std_top = np.std([offspring[i][0] for i in range(mu)])
            if std_top > 1e-15:
                sigma *= np.exp(c_sigma / d_sigma * (std_top / (old_sigma + 1e-10) - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Simple covariance update (rank-1 approximation)
            # This is a simplified version; full CMA-ES would be more complex
            diff = new_x_mean - x_mean
            C = (1 - c_1 - c_mu) * C + c_1 * (1 + (1 - c_sigma) * c_sigma) * np.outer(diff, diff)
            C = (C + C.T) / 2  # Symmetrize
            
            x_mean = new_x_mean
            generation += 1
            
            # Check for convergence
            if std_top < 1e-10:
                break
        
        # Phase 2: Local polish with L-BFGS-B
        if time_ok() and calls < budget - 10:
            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)
            
            # Try L-BFGS-B
            try:
                res_lbfgs = minimize(
                    local_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res_lbfgs.fun < best_fx:
                    best_fx = res_lbfgs.fun
                    best_x = np.asarray(res_lbfgs.x, dtype=np.float64).copy()
            except Exception:
                pass
            
            # If L-BFGS-B didn't help much, try Nelder-Mead
            if time_ok() and calls < budget - 10:
                try:
                    res_nm = minimize(
                        local_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if res_nm.fun < best_fx:
                        best_fx = res_nm.fun
                        best_x = np.asarray(res_nm.x, dtype=np.float64).copy()
                except Exception:
                    pass
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
