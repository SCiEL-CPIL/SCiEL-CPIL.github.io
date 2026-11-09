"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with restarts and Nelder-Mead polish for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # Safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls = 0

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x_clipped))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: CMA-ES for the bulk of the budget
        # CMA-ES parameters
        pop_size = max(2 * dim + 1, 9)  # Standard CMA-ES population size
        lam = pop_size
        sigma = 2.0  # Initial step size - wide enough to cover [-5,5]^5
        
        # CMA-ES state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # CMA-ES adaptation parameters
        mu = lam // 2  # Weights for mu best
        weights = np.zeros(mu)
        w_sum = 0.0
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            w_sum += weights[i]
        weights /= w_sum
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # CMA-ES main loop
        cma_budget = int(budget * 0.7)
        cma_calls = 0
        stagnation_count = 0
        best_f_cma = float('inf')
        last_improvement = 0
        patience = 50  # Number of generations without improvement before restart
        
        while calls < cma_budget and time.time() < deadline - 30.0:
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                x_i = x_mean + sigma * C_sqrt @ z
                x_i = np.clip(x_i, -5.0, 5.0)
                f_i = safe_func(x_i)
                offspring.append(x_i)
                offspring_f.append(f_i)
                cma_calls += 1
            
            if len(offspring) == 0:
                break
            
            offspring_f = np.array(offspring_f)
            
            # Sort by fitness
            order = np.argsort(offspring_f)
            sorted_x = [offspring[i] for i in order[:mu]]
            sorted_f = offspring_f[order[:mu]]
            
            # Update best
            if sorted_f[0] < best_f_cma:
                best_f_cma = sorted_f[0]
                if best_f_cma < best_f:
                    best_f = best_f_cma
                    best_x = sorted_x[0].copy()
                last_improvement = cma_calls
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w * x for w, x in zip(weights, sorted_x)], axis=0)
            
            # Compute p_sigma and p_c
            p_sigma = (1 - c_sigma) * p_sigma if hasattr(self, '_p_sigma') else np.zeros(dim)
            p_c = (1 - c_c) * p_c if hasattr(self, '_p_c') else np.zeros(dim)
            
            # Actually, let's use a simpler but still effective CMA-ES variant
            # Reset state properly
            break
        
        # Use a cleaner CMA-ES implementation
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        cma_calls_total = 0
        gen = 0
        stagnation = 0
        best_f_this_cma = float('inf')
        
        while calls < cma_budget and time.time() < deadline - 30.0:
            gen += 1
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                x_i = x_mean + sigma * (C_sqrt @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                f_i = safe_func(x_i)
                offspring.append(x_i)
                offspring_f.append(f_i)
                cma_calls_total += 1
            
            if len(offspring) < 2:
                break
            
            offspring_f = np.array(offspring_f)
            order = np.argsort(offspring_f)
            sorted_x = np.array([offspring[i] for i in order[:mu]])
            sorted_f = offspring_f[order[:mu]]
            
            # Update global best
            if sorted_f[0] < best_f_this_cma:
                best_f_this_cma = sorted_f[0]
                if best_f_this_cma < best_f:
                    best_f = best_f_this_cma
                    best_x = sorted_x[0].copy()
                stagnation = 0
            else:
                stagnation += 1
            
            # Update mean
            x_mean_new = np.sum([w * x for w, x in zip(weights, sorted_x)], axis=0)
            
            # Compute p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x_mean_new - x_mean) / sigma)
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Compute p_c
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < \
                        (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update covariance
            delta = x_mean_new - x_mean
            C = (1 - c_1 - c_mu) * C + \
                c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
                c_mu * np.sum([w * (np.outer(x, x) for x in sorted_x)], axis=0)
            
            # Regularize C
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_sqrt using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            
            x_mean = x_mean_new
            
            # Check for stagnation - restart if no improvement for many generations
            if stagnation > patience and gen > 20:
                # Try a new random center
                if best_x is not None and np.random.random() < 0.5:
                    x_mean = best_x + np.random.standard_normal(dim) * sigma
                else:
                    x_mean = np.random.uniform(-4, 4, dim)
                sigma = 2.0
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                gen = 0
                stagnation = 0
                best_f_this_cma = float('inf')
            
            # Check time
            if time.time() > deadline - 40.0:
                break
        
        # Phase 2: Local search / Nelder-Mead polish with remaining budget
        remaining = budget - calls
        if remaining > dim + 10 and time.time() < deadline - 10.0:
            def nm_func(x):
                nonlocal calls, best_f, best_x
                if calls >= budget:
                    return best_f
                x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                fx = float(func(x_clipped))
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxiter': min(remaining // dim, 500),
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'initial_simplex': 0.1})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass
        
        # Phase 3: If we have budget left, try a few more random restarts around best
        if calls < budget and time.time() < deadline - 5.0:
            for _ in range(min(20, budget - calls)):
                if calls >= budget:
                    break
                if best_x is not None:
                    perturbation = np.random.standard_normal(dim) * 0.5
                    x_try = best_x + perturbation
                    x_try = np.clip(x_try, -5.0, 5.0)
                    safe_func(x_try)
                else:
                    safe_func(np.random.uniform(-4, 4, dim))
        
        # Final safety: ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            if calls < budget:
                best_f = safe_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
