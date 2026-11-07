"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            return val

        # Phase 1: CMA-ES with restarts
        # Budget allocation: use most of the budget for CMA-ES
        cma_budget = self.budget - 50  # reserve 50 for local refinement
        
        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * self.dim)  # population size
        c_sigma = (lambda_ + 2) / (self.dim + lambda_ + 5)
        c_c = (4 + lambda_ / self.dim) / (self.dim + 4 + 2 * lambda_ / self.dim)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = (2 / ((self.dim + 1.3) ** 2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((self.dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        for restart in range(5):
            if calls_made >= cma_budget or time.time() > deadline:
                break
                
            # Initialize
            if best_x is not None and restart > 0:
                x = best_x + rng.normal(0, 0.5, self.dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-4, 4, self.dim)
            
            fx = safe_call(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            sigma = 2.0 if restart == 0 else 0.5
            B = np.eye(self.dim)
            D = np.ones(self.dim)
            ps = np.zeros(self.dim)
            pc = np.zeros(self.dim)
            
            # Generate initial population
            pop = []
            pop_f = []
            for i in range(lambda_):
                if calls_made >= cma_budget or time.time() > deadline:
                    break
                z = rng.standard_normal(self.dim)
                xi = x + sigma * (B @ (D * z))
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                pop.append(xi)
                pop_f.append(fi)
            
            if len(pop) < mu:
                continue
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in indices]
            pop_f_sorted = [pop_f[i] for i in indices]
            
            # Update best
            if pop_f_sorted[0] < best_f:
                best_f, best_x = pop_f_sorted[0], pop_sorted[0].copy()
            
            # Weighted mean
            x_new = sum(weights[i] * pop_sorted[i] for i in range(mu))
            
            # Update covariance matrix
            y = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < (1.4 + 2 / (self.dim + 1))
            pc = (1 - c_c) * pc + hsig * c_c * y
            
            C = B @ (D ** 2) @ B.T
            C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(ps, ps))
            C += c_mu * sum(weights[i] * np.outer(pop_sorted[i] - x, pop_sorted[i] - x) / sigma ** 2 for i in range(mu))
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Eigen decomposition for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-10))
            B = eigvecs
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            
            x = x_new
        
        # Phase 2: Local refinement with Nelder-Mead
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            from scipy.optimize import minimize
            
            def wrapped_func(x):
                if time.time() > deadline or calls_made >= self.budget:
                    return np.inf
                return safe_call(x)
            
            try:
                result = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < best_f:
                    best_f, best_x = result.fun, result.x
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
