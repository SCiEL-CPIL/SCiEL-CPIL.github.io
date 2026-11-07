"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: 4 + floor(3 * log(dim))
        lam = max(4, int(4 + 3 * np.log(dim)))
        # For dim=5, lam = 4 + 3*1.609 = 8.8 -> 8. Let's use 9 or 10 for stability.
        lam = 9
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial values
        x_mean = np.zeros(dim) # Start at center
        sigma = 2.0 # Initial step size, box is [-5, 5]
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            calls_made += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Evaluate initial mean
        f_mean = safe_eval(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        # Main Loop
        while calls_made < budget:
            # Generate population
            y = np.array([rng.multivariate_normal(np.zeros(dim), np.eye(dim)) for _ in range(lam)])
            # Transform to x
            x_pop = np.array([x_mean + sigma * (B @ (D * y[i])) for i in range(lam)])
            
            # Evaluate population
            f_pop = np.array([safe_eval(x_pop[i]) for i in range(lam)])
            
            # Check if we ran out of budget
            if calls_made >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            # Weighted mean
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update path states
            pc = (1 - cc) * pc + cc * (x_new - x_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * (x_new - x_mean) / sigma))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made + 1) / (dim + mueff / 2))) < (1.4 + 2 / (dim + 1)) * (1 - cs))
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (np.outer(pc, pc) - (1 - cc) * C) + cmu * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - x_mean, x_sorted[:mu] - x_mean) / sigma**2, axis=0)
            
            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            # Update step size (Cumulative Step Size Adaptation)
            sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made + 1) / (dim + mueff / 2))) - 1))
            
            # Update mean
            x_mean = x_new
            
            # Check termination conditions (optional, but good for efficiency)
            if sigma < 1e-12 or np.max(D) / np.min(D) > 1e14:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
