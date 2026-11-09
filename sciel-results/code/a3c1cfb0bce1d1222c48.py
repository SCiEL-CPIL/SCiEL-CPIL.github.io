"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))  # Population size

        # Initial population centered at origin, spread across the box
        x0 = np.zeros(n)
        f0 = func(x0)
        
        # Initial step size: large enough to explore, small enough to converge
        sigma = 1.0 
        
        # CMA-ES parameters
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        x_mean = x0.copy()
        C = np.eye(n)
        C_sqrt = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (20 * n**2))
        
        best_x = x0.copy()
        best_f = f0
        calls_used = 1
        
        for gen in range(1, self.budget // lam + 1):
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(lam):
                # Cholesky factor update is usually done per generation, 
                # but for simplicity and speed in this context, we update C_sqrt less frequently 
                # or use a simplified eigenvalue update if needed. 
                # Standard CMA-ES updates C every generation.
                # To save time, we will update C_sqrt every few generations or use a cheaper method.
                # However, for correctness, let's do the standard update.
                # Note: np.linalg.cholesky can be expensive. 
                # For n=5, it's fast.
                
                z = rng.standard_normal(n)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                population.append((y, i))
            
            # Evaluate
            for y, i in population:
                if time.time() > deadline:
                    break
                fy = func(y)
                calls_used += 1
                population[i] = (y, fy)
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Update best
            if population[0][1] < best_f:
                best_f = population[0][1]
                best_x = population[0][0].copy()
            
            # Selection
            y_w = np.array([p[0] for p in population[:mu]])
            f_w = np.array([p[1] for p in population[:mu]])
            
            # New mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * y_w, axis=0)
            
            # Update step size (CSA)
            if np.linalg.norm(C_sqrt) > 1e-16:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ ((x_mean - x_mean_old) / (sigma * np.linalg.norm(C_sqrt))))
                h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
                if h_sigma < 1.4:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # Rank-mu update
            v = (y_w - x_mean_old) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(n)) + c_mu * np.sum(w[:, None, None] * np.outer(v, v), axis=0)
            
            # Rank-one update for pc
            pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma
            
            # Ensure positive definiteness and update square root
            # Simple repair: add small diagonal if needed
            C += 1e-12 * np.eye(n)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                C_sqrt = vecs @ np.diag(np.sqrt(vals))
                
        return best_x, best_f
# EVOLVE-BLOCK-END
