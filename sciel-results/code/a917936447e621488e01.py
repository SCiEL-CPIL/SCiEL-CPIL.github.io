"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: coarse random/grid search followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse random search to find a good starting point
        # Use a mix of uniform random samples and some structured grid points
        # Allocate about 20% of the budget to exploration
        n_explore = max(10, min(50, budget // 5))
        
        # Generate diverse starting points
        candidates = []
        
        # Uniform random points
        for _ in range(n_explore // 2):
            candidates.append(rng.uniform(-5, 5, dim))
        
        # Structured grid points (coarse)
        # For dim=5, we can't do a full grid, so use stratified sampling
        for _ in range(n_explore // 2):
            # Each coordinate is in one of 3 ranges: low, mid, high
            x = np.empty(dim)
            for i in range(dim):
                zone = rng.integers(0, 3)
                if zone == 0:
                    x[i] = rng.uniform(-5, -1.67)
                elif zone == 1:
                    x[i] = rng.uniform(-1.67, 1.67)
                else:
                    x[i] = rng.uniform(1.67, 5)
            candidates.append(x)
        
        # Evaluate all candidates
        best_x = None
        best_f = np.inf
        for x in candidates:
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES starting from the best point found
        x = best_x.copy()
        fx = best_f
        remaining = budget - len(candidates)
        
        # Initialize CMA-ES parameters
        sigma = 1.0
        # Adaptive step size based on dimension
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
        cmu = (2 + 2 / (dim + 1) - dim * c1) / ((dim + 1.3) ** 2 + 2)
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # CMA-ES state
        mean = x.copy()
        C = np.eye(dim)
        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Population size
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        
        for iteration in range(remaining):
            if iteration >= remaining:
                break
            
            # Generate population
            offspring = []
            for _ in range(lambda_pop):
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
            
            offspring_f = np.array(offspring_f)
            sorted_idx = np.argsort(offspring_f)
            
            # Update mean
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[sorted_idx[i]]
            
            # Update step size
            # Calculate distance
            d = np.linalg.norm(new_mean - mean) / (sigma * np.sqrt(C))
            # Simplified: use norm
            diff = new_mean - mean
            # Approximate using C
            try:
                C_inv = np.linalg.inv(C)
                d_sq = diff @ C_inv @ diff
                d = np.sqrt(max(0, d_sq))
            except np.linalg.LinalgError:
                d = np.linalg.norm(diff) / sigma
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C @ diff) / (sigma * chi_n) if d > 0 else (1 - cs) * ps
            # Simplified ps update
            if d > 1e-10:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff / (sigma * chi_n)
            
            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            
            # Update C
            for i in range(mu):
                y_i = (offspring[sorted_idx[i]] - mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (diff @ diff.T + (1 - hsig) * cs * (2 - cs) * ps @ ps.T) + cmu * (weights[i] * (y_i @ y_i.T))
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10)
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Track best
            best_off_f = offspring_f[sorted_idx[0]]
            if best_off_f < fx:
                fx = best_off_f
                x = offspring[sorted_idx[0]].copy()
        
        return x, fx
# EVOLVE-BLOCK-END
