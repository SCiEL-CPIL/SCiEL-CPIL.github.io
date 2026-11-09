"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES style optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 325.0  # 300s limit, 25s margin
        self.dim = self.dim
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        n = self.dim
        rng = np.random.default_rng(42)
        
        # Initial population size
        lambda_pop = max(4 + int(3 * np.log(n)), 8)
        
        # Initial mean: try zero first, then random
        mean = np.zeros(n)
        
        # Initial sigma: start wide to explore
        sigma = 2.0
        
        # CMA-ES parameters
        mu = lambda_pop // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - 1 / mu_eff + c_sigma) / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State variables
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = mean.copy()
        best_fx = float('inf')
        
        # First evaluation
        try:
            best_fx = eval_func(best_x)
        except StopIteration:
            return best_x, best_fx
        
        generation = 0
        while self.calls < self.budget and time.time() - self.start_time < self.time_limit:
            # Generate offspring
            xs = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                xs[i] = mean + sigma * (B @ (D * z))
            
            # Evaluate offspring
            fs = np.full(lambda_pop, float('inf'))
            for i in range(lambda_pop):
                try:
                    fs[i] = eval_func(xs[i])
                except StopIteration:
                    break
            
            if np.all(np.isinf(fs)):
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_fx:
                best_fx = fs[0]
                best_x = xs[0].copy()
            
            # Update mean
            new_mean = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Update evolution paths
            old_mean = mean
            mean = new_mean
            
            # Update cs
            cs = np.sqrt(1 - 2 * c_sigma * (1 + d_sigma * c_sigma))
            ps = (1 - c_sigma) * ps + cs * (B @ (D * (mean - old_mean) / sigma))
            
            # Update C
            c = (1 - c_c) * C
            for i in range(mu):
                c += weights[i] * np.outer(xs[i] - old_mean, xs[i] - old_mean) / (sigma ** 2)
            
            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(c)
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            C = B @ (D[:, None] * (D[None, :] * B.T))
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            generation += 1
            
            # Check if we've converged
            if sigma < 1e-8:
                break
            
            # Periodic restart if stuck
            if generation % 50 == 0 and sigma < 0.01:
                # Try a small random perturbation
                mean = best_x + rng.standard_normal(n) * 0.1
                sigma = 0.5
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
        
        # Final check
        if not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if self.calls < self.budget:
                try:
                    best_fx = eval_func(best_x)
                except StopIteration:
                    pass
            else:
                best_fx = 0.0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
