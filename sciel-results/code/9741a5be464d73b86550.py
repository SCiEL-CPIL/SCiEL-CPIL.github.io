"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: LHS scan + CMA-ES refinement for 5D Sphere."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        # Phase 1: Latin Hypercube Sampling to quickly locate the basin
        # Use ~30% of budget for initialization scan
        n_scan = min(150, budget // 3)
        
        # Generate LHS points in [-4, 4]^dim (where minimum is guaranteed)
        n_per_dim = n_scan
        lhs_points = np.zeros((n_scan, dim))
        for d in range(dim):
            perm = rng.permutation(n_per_dim)
            # Map to [-4, 4]
            lhs_points[:, d] = -4.0 + 8.0 * (perm + rng.random(n_per_dim)) / n_per_dim
        
        # Evaluate all scan points
        best_x = None
        best_f = np.inf
        for i in range(n_scan):
            if time.time() > deadline:
                break
            x = np.clip(lhs_points[i], -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Also evaluate center as a baseline
        x_center = np.zeros(dim)
        f_center = func(x_center)
        if f_center < best_f:
            best_f = f_center
            best_x = x_center.copy()
        
        calls_used = n_scan + 1
        remaining = budget - calls_used
        
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES refinement from the best point found
        # Initialize CMA-ES with a small sigma around the best point
        x0 = best_x.copy()
        sigma0 = 1.0  # Start with moderate step size
        
        # CMA-ES parameters for 5D
        # Population size: 4 + floor(3 * log(5)) ≈ 4 + 5 = 9
        lambda_ = 9
        mu = lambda_ // 2  # Number of parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialize state
        mean = x0.copy()
        C = np.eye(dim)
        Cs = np.zeros(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = sigma0
        cs_sqrt = np.sqrt(cs * (dim + 2) * cs)
        c1_sqrt = np.sqrt(c1)
        cmu_sqrt = np.sqrt(cmu)
        
        # Precompute Cholesky factor
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(dim)
        
        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        calls_made = calls_used
        max_iter = 50
        
        for iteration in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                try:
                    y = mean + sigma * (B @ z)
                except:
                    y = mean + sigma * z
                pop[i] = np.clip(y, -5, 5)
            
            # Evaluate population
            fit = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    # Fill remaining with best known
                    fit[i] = global_best_f
                    continue
                fit[i] = func(pop[i])
                calls_made += 1
                if fit[i] < global_best_f:
                    global_best_f = fit[i]
                    global_best_x = pop[i].copy()
            
            # Sort by fitness
            order = np.argsort(fit)
            pop_sorted = pop[order]
            fit_sorted = fit[order]
            
            # Update mean
            mean = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Update evolution paths
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except:
                C_inv_sqrt = np.eye(dim)
            
            dmean = (mean - (mean - sigma * cs_sqrt * Cs)) / sigma
            pc = (1 - cc) * pc + cc * C_inv_sqrt @ dmean
            norm_pc = np.linalg.norm(pc)
            hsig = norm_pc / np.sqrt(1 - (1 - cc) ** (2 * (iteration + 1) / (dim + 1))) / np.sqrt(dim) < (1 + 2 / (dim + 1)) * np.sqrt(cc + 1)
            
            ps = (1 - cs) * ps + cs_sqrt * hsig * dmean
            norm_ps = np.linalg.norm(ps)
            
            # Update covariance
            delta_h = (1 - hsig) * cc * (1 - cc)
            C = (1 - c1 - cmu + delta_h) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_sorted[:mu] - mean, pop_sorted[:mu] - mean) for _ in [0])
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 2) * (norm_ps / np.sqrt(dim) - 1))
            
            # Renormalize C
            try:
                L = np.linalg.cholesky(C)
                B = L
            except:
                # Regularize
                C += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    B = L
                except:
                    B = np.eye(dim)
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # If no progress in last few iterations, break
            if iteration > 10 and global_best_f > best_f - 1e-6 and iteration > 20:
                break
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
