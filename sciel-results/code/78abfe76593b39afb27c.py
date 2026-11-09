"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Coarse Grid Initialization for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # 1. Coarse Grid Search to find a good starting point and ensure early progress
        # We use a sparse grid to conserve budget for the main optimizer.
        # 3 points per dimension -> 3^5 = 243 points. This is a significant chunk of 500,
        # but it guarantees we find the basin if it's not extremely narrow/hidden.
        # However, 243 calls leaves only 257 for CMA-ES. 
        # Let's use a sparser grid or a smarter initialization.
        # Rosenbrock minimum is usually near 1 in each dimension for standard orientation.
        # With rotation, it's shifted.
        # Let's try a "Latin Hypercube" or just a few strategic points + random.
        # Actually, the directive suggests a "deterministic grid-based global search".
        # Let's do a very coarse grid: 2 points per dim? 2^5 = 32 points.
        # Or maybe 3 points per dim is too much.
        # Let's stick to the directive's spirit but be careful with budget.
        # A 3x3x3x3x3 grid is 243 calls. That's 48% of the budget.
        # If we do that, we have 257 calls left. CMA-ES with lambda=9 (dim*2+1) takes ~257/9 ~ 28 generations.
        # That might be enough for Rosenbrock if started well.
        
        # Alternative: Use a smaller grid. 2 points per dimension (corners + center?) No, grid implies lattice.
        # Let's use 3 points per dimension: -3, 0, 3.
        # This covers a large part of the box.
        
        grid_points = []
        # Generate grid points in [-3, 3] to stay safely inside [-5, 5] and cover likely area
        # Actually, the minimum is in [-4, 4]. Let's use -4, 0, 4.
        vals = [-4.0, 0.0, 4.0]
        
        import itertools
        for combo in itertools.product(vals, repeat=dim):
            grid_points.append(np.array(combo))
            
        # Evaluate grid points
        best_grid_x = None
        best_grid_f = np.inf
        
        for p in grid_points:
            if time.time() > deadline or calls_used >= budget:
                break
            p_clipped = np.clip(p, -5, 5)
            f_p = call_func(p_clipped)
            if f_p < best_grid_f:
                best_grid_f = f_p
                best_grid_x = p_clipped.copy()

        # 2. CMA-ES Optimization
        if best_grid_x is not None and calls_used < budget:
            # Initialize CMA-ES parameters
            x0 = best_grid_x
            sigma0 = 1.0 # Initial step size, adaptive
            
            # CMA-ES implementation
            # Population size lambda = 4 + floor(3 * log(n))
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            # CMA-ES parameters
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cs
            
            # State
            x_mean = x0.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            sigma = sigma0
            
            # Track best
            best_cma_x = x_mean.copy()
            best_cma_f = best_grid_f
            
            while time.time() < deadline and calls_used < budget:
                # Check if we have enough budget for a full population
                if calls_used + lam > budget:
                    break
                
                # Generate offspring
                offspring = []
                f_offspring = []
                
                for i in range(lam):
                    # Sample from multivariate normal
                    z = np.random.normal(0, 1, dim)
                    # Use Cholesky decomposition of C for sampling
                    # C = C_sqrt @ C_sqrt.T
                    # We need to update C_sqrt periodically, but for simplicity and speed,
                    # we can use the fact that C is symmetric positive definite.
                    # For dim=5, Cholesky is cheap.
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to identity if C becomes singular
                        C_sqrt = np.eye(dim)
                    
                    y = C_sqrt @ z
                    x_i = x_mean + sigma * y
                    x_i = np.clip(x_i, -5, 5)
                    
                    f_i = call_func(x_i)
                    offspring.append(x_i)
                    f_offspring.append(f_i)
                
                # Sort offspring by fitness
                indices = np.argsort(f_offspring)
                offspring = [offspring[i] for i in indices]
                f_offspring = [f_offspring[i] for i in indices]
                
                # Update best
                if f_offspring[0] < best_cma_f:
                    best_cma_f = f_offspring[0]
                    best_cma_x = offspring[0].copy()
                    # Update global best
                    if best_cma_f < best_f:
                        best_f = best_cma_f
                        best_x = best_cma_x.copy()

                # Update mean
                x_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
                
                # Update evolution path for covariance matrix adaptation
                C_old = C
                C_sqrt_inv = np.linalg.inv(C_sqrt)
                y_bar = (x_new - x_mean) / sigma
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ y_bar
                norm_ps = np.linalg.norm(ps) / np.sqrt(np.trace(C)) # Approximation, actually norm(ps)/sigma? No, norm(ps) is unitless in C space?
                # Standard CMA-ES: hsig = (norm(ps) / sqrt(1 - (1 - cs)^{2*(gen+1)})) / (norm(C_sqrt_inv @ e) * sigma)
                # Simplified hsig check:
                hsig = (norm_ps / np.sqrt(np.trace(C))) < (1.4 + 2 / (dim + 1)) * (1 - cs) # Rough heuristic
                
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_bar
                
                # Update Covariance
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                rank_one = np.outer(y_bar, y_bar)
                rank_mu = np.sum([weights[i] * np.outer(offspring[i] - x_mean, offspring[i] - x_mean) for i in range(mu)], axis=0) / (sigma * sigma)
                
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C) + cmu * rank_mu
                
                # Ensure C is symmetric and positive semi-definite
                C = (C + C.T) / 2
                # Clip eigenvalues to be positive
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(np.trace(C)) - 1))
                
                # Update mean
                x_mean = x_new
                
                # Check for stagnation
                if np.std(f_offspring[:mu]) < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
