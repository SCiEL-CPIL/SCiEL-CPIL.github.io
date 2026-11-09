"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic global-to-local hierarchical search: coarse grid then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse deterministic grid scan (~400 calls)
        # Use 3^5 = 243 grid points at {-5, 0, 5}^5
        # Plus a second layer of 5^2 * 3^3 = 675 is too many; use 4^3 * 3^2 = 432
        # Actually let's use 3^5=243 + a refined 5-point grid in top region
        # Simpler: 3^5 = 243 points, then use remaining for CMA-ES
        
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append([i * 5 - 5, j * 5 - 5, k * 5 - 5,
                                                 l * 5 - 5, m * 5 - 5])
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        for point in grid_points:
            if calls_used >= budget:
                break
            x = np.array(point, dtype=float)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1
        
        # Phase 2: CMA-ES with proper Cholesky-based sampling
        remaining = budget - calls_used
        if remaining < 10:
            return best_x, best_f
        
        # CMA-ES parameters
        x = best_x.copy()
        sigma = 2.0  # Start with larger step to explore local basin
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        cs = 1.0 / (dim + 4)
        c1 = 2.0 / ((dim + 1.3) ** 2 + 10)
        cmu = min(1 - c1, 2 * ((dim - 2) + 2 / (dim + 1)) / ((dim + 2) ** 2 + 15))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Population size
        lam = max(4, min(4 * dim + 2, remaining // 3))
        mu = max(1, lam // 2)
        
        # Weights for recombination
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Cholesky factor of C
        L = np.linalg.cholesky(C)
        
        # Deterministic RNG seed for reproducibility
        rng = np.random.default_rng(42)
        
        gen = 0
        while calls_used + lam <= budget:
            gen += 1
            # Sample offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
                calls_used += 1
            
            if len(offspring) < mu:
                break
            
            # Sort
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Weighted mean
            new_mean = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            
            # Update paths
            d_mean = (new_mean - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * d_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen))) < \
                   (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * d_mean
            
            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x)
            
            # Ensure C is positive definite
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(np.eye(dim) + 1e-10 * np.eye(dim))
                C = np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            x = new_mean
            
            # If sigma is very small, stop
            if sigma < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
