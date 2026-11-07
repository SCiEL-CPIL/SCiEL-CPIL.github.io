"""Rewritten optimizer using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f17 (Schaffers F7) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5, 5)

        # CMA-ES parameters
        # Initial mean: center of the box
        m = np.zeros(n)
        # Initial sigma: large enough to explore [-5, 5]^5
        # A common heuristic is (b_max - b_min) / 6 or similar.
        # For n=5, sigma=2.0 covers most of the space.
        sigma = 2.0
        # Population size
        # Standard rule: lambda = 4 + floor(3 * ln(n))
        # For n=5, ln(5)~1.6, 3*1.6=4.8, floor=4, lambda=8.
        # Let's use lambda=10 for better exploration in early stages.
        lambda_pop = 10
        
        # CMA-ES internal parameters
        # Learning rates
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (n + 2)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Eigenvalue decay
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State variables
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        step_size = sigma
        step_size_success = 1.0 # Placeholder for log updates
        
        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_f, best_x
            x = clip(x)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation of mean? No, CMA-ES samples around mean.
        # We need to ensure we don't exceed budget.
        
        while calls + lambda_pop <= budget:
            # Sample population
            pop = []
            for i in range(lambda_pop):
                # Cholesky decomposition for sampling
                # In a simple implementation, we can use rng.multivariate_normal
                # but for performance and stability, we might want to update C explicitly.
                # Given the small dimension (5), multivariate_normal is fine.
                try:
                    x = rng.multivariate_normal(m, C * step_size**2)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    x = m + step_size * rng.normal(0, 1, n)
                pop.append(x)
            
            # Evaluate population
            pop_f = [evaluate(x) for x in pop]
            
            if calls >= budget:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update mean
            new_m = sum(weights[i] * pop[i] for i in range(mu))
            
            # Update step size (sigma)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / step_size * (C^(-1/2) * (new_m - m))
            # To avoid explicit matrix square root inversion, we can approximate or use a simpler update.
            # For n=5, we can compute the square root.
            try:
                # Compute C^(-1/2)
                w, v = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                w = np.maximum(w, 1e-10)
                C_inv_sqrt = v @ np.diag(1 / np.sqrt(w)) @ v.T
                diff = new_m - m
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / step_size * (C_inv_sqrt @ diff)
                sigma_norm = np.linalg.norm(ps)
                h_sigma = 1.0 if sigma_norm / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / budget)) < (1 + 2 / (n + 1)) * chi_n else 0.0
                step_size *= np.exp(c_sigma / d_sigma * (sigma_norm / chi_n - 1))
            except Exception:
                # Fallback simple sigma update
                step_size *= 0.9 if h_sigma == 0 else 1.1
                h_sigma = 0.0

            # Update covariance matrix
            # pc = (1 - c_c) * pc + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (new_m - m) / step_size
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_m - m) / step_size
            
            # delta_h = h_sigma / c_c - 1 + (1 - h_sigma) * (1 - c_c)
            delta_h = h_sigma / c_c - 1 + (1 - h_sigma) * (1 - c_c)
            
            # C = (1 - c_1 - c_mu + delta_h * c_c * (2 - c_c)) * C + c_1 * pc * pc.T
            C = (1 - c_1 - c_mu + delta_h * c_c * (2 - c_c)) * C + c_1 * np.outer(pc, pc)
            
            # + c_mu * sum(weights[i] * pop[i] * pop[i].T)
            diff_pop = [pop[i] - m for i in range(mu)]
            C += c_mu * sum(weights[i] * np.outer(diff_pop[i], diff_pop[i]) for i in range(mu))
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if necessary
            # For n=5, adding a small identity if eigenvalues are too small
            try:
                w, _ = np.linalg.eigh(C)
                if np.min(w) < 1e-10:
                    C += 1e-10 * np.eye(n)
            except Exception:
                C = np.eye(n)
            
            m = new_m

        return best_x, best_f
# EVOLVE-BLOCK-END
