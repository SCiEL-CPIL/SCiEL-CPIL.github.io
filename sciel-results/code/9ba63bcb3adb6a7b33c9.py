"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with deterministic coarse grid initialization for BBOB f4.
    
    Strategy:
    1. Deterministic coarse grid scan to robustly locate the global basin
    2. CMA-ES to adapt covariance to the rotated landscape and converge
    3. Final polish with small steps
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        def time_check():
            return time.time() > deadline

        # Phase 1: Deterministic coarse grid scan
        # Büche-Rastrigin is multimodal; we need to find the global basin first.
        # Use a stratified grid over [-4, 4]^5 (minimizer is in [-4,4]^5)
        
        # Allocate budget: ~40% for grid, ~50% for CMA-ES, ~10% for polish
        grid_budget = int(self.budget * 0.4)
        cma_budget = int(self.budget * 0.5)
        polish_budget = self.budget - grid_budget - cma_budget

        # Generate stratified grid points
        # For 5D, we can't do a full grid, so use a quasi-random stratified sampling
        n_grid = min(grid_budget, 243)  # 243 = 3^5, a reasonable grid size
        
        # Create a deterministic stratified sample
        # Divide each dimension into 3 bins, sample one point per bin
        # This gives us 3^5 = 243 points
        grid_points = []
        if n_grid >= 243:
            # Full 3^5 stratified grid
            for i1 in range(3):
                for i2 in range(3):
                    for i3 in range(3):
                        for i4 in range(3):
                            for i5 in range(3):
                                x = np.array([
                                    -4 + (i1 + 0.5) * (8.0 / 3),
                                    -4 + (i2 + 0.5) * (8.0 / 3),
                                    -4 + (i3 + 0.5) * (8.0 / 3),
                                    -4 + (i4 + 0.5) * (8.0 / 3),
                                    -4 + (i5 + 0.5) * (8.0 / 3)
                                ])
                                grid_points.append(x)
        else:
            # If budget is smaller, use random stratified sampling
            for _ in range(n_grid):
                if time_check():
                    break
                x = rng.uniform(-4, 4, self.dim)
                grid_points.append(x)
        
        # Evaluate all grid points
        for x in grid_points:
            if calls >= self.budget or time_check():
                break
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from the best point found
        if best_x is not None and calls < self.budget and not time_check():
            x = best_x.copy()
            fx = best_f
            
            # CMA-ES parameters
            dim = self.dim
            popsize = max(4 + int(3 * np.log(dim)), 8)  # Standard CMA-ES population size
            popsize = min(popsize, cma_budget // 2)  # Don't use more than half the CMA budget
            popsize = max(popsize, 4)  # Minimum population size
            
            # Initial step size
            sigma = 1.0
            sigma_min = 0.001
            sigma_max = 5.0
            
            # CMA-ES state
            mean = x.copy()
            C = np.eye(dim)  # Covariance matrix
            C_sqrt = np.eye(dim)  # Square root of covariance
            pc = np.zeros(dim)  # Evolution path for covariance
            ps = np.zeros(dim)  # Evolution path for step size
            cs = (4 + dim / 2) / ((dim + 4) * 3 + 3)  # Step size adaptation rate
            ds = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs  # Damping for sigma
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) * (dim + 2) + 2)  # Covariance adaptation rate
            cmu = min(1 - c1, 2 * (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2))  # Covariance adaptation rate
            cc = (4 + dim / dim) / (dim + 4)  # Evolution path rate for covariance
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))  # Expected ||N(0,I)||
            
            # Count CMA-ES iterations
            cma_calls_used = 0
            max_cma_iters = cma_budget // popsize + 1
            
            for iteration in range(max_cma_iters):
                if calls >= self.budget or time_check():
                    break
                
                # Generate population
                offspring = []
                offspring_f = []
                
                for i in range(popsize):
                    if calls >= self.budget or time_check():
                        break
                    
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    
                    fy = evaluate(y)
                    if fy is None:
                        break
                    
                    offspring.append(y)
                    offspring_f.append(fy)
                    cma_calls_used += 1
                
                if len(offspring) < popsize:
                    break
                
                # Sort offspring by fitness
                indices = np.argsort(offspring_f)
                sorted_offspring = [offspring[i] for i in indices]
                sorted_f = [offspring_f[i] for i in indices]
                
                # Update best
                if sorted_f[0] < best_f:
                    best_f = sorted_f[0]
                    best_x = sorted_offspring[0].copy()
                
                # Update mean (weighted average of top half)
                mu = popsize // 2
                weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
                weights /= weights.sum()
                
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * sorted_offspring[i]
                
                # Update evolution paths
                old_mean = mean.copy()
                mean = new_mean
                
                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt.T @ ((mean - old_mean) / sigma))
                mu_eff = (weights * weights).sum()
                
                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * ((mean - old_mean) / sigma)
                
                # Update covariance
                delta_sig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chiN else 0.0
                C = (1 + c1 * delta_sig - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * sum(w * np.outer(y - old_mean, y - old_mean) for w, y in zip(weights, sorted_offspring[:mu]))
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Fallback: add small regularization
                    C += 1e-10 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C_sqrt = np.eye(dim)
                
                # Update sigma
                sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Check for convergence
                if sigma < sigma_min or np.trace(C) / dim < sigma_min ** 2:
                    break
        
        # Phase 3: Final polish with very small steps
        if best_x is not None and calls < self.budget and not time_check():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01  # Very small sigma for fine-tuning
            polish_steps = min(polish_budget, 30)
            
            for step in range(polish_steps):
                if calls >= self.budget or time_check():
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.9, 0.001)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
