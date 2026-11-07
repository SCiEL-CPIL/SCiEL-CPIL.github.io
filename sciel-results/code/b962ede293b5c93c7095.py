"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Two-stage optimizer: Sobol-based exploration followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_points(self, n, dim):
        """Generate n Sobol points in [0, 1]^dim."""
        # Simple implementation of Sobol sequence for low dimensions
        # For simplicity and robustness, we use a basic halving or pre-computed logic
        # However, standard numpy doesn't have Sobol. We can use a simple quasi-random approach
        # or implement a basic version. Given the constraints, let's use a simple stratified sampling
        # which is similar in spirit to Sobol for low dimensions.
        
        # Actually, let's implement a simple Sobol-like generator for small dimensions
        # Or just use Latin Hypercube Sampling which is easy and effective.
        
        # LHS:
        # Divide [0,1] into n intervals, pick one point from each in each dimension
        points = np.zeros((n, dim))
        for d in range(dim):
            # Shuffle the order of intervals
            perm = np.random.permutation(n)
            # For each point i, it falls into interval perm[i]
            # Uniform within that interval
            points[:, d] = (perm + np.random.rand(n)) / n
        return points

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Stage 1: Dense stratified grid search (LHS) for the first 50 calls
        n_grid = 50
        if budget < n_grid:
            n_grid = budget
            
        # Generate LHS points in [-5, 5]^dim
        # Map [0,1] to [-5,5]
        lhs_unit = self._sobol_points(n_grid, dim)
        lhs_points = -5 + 10 * lhs_unit
        
        x_best = None
        f_best = np.inf
        calls = 0
        
        # Evaluate grid points
        for i in range(n_grid):
            if time.time() - start_time > time_limit:
                break
            x = lhs_points[i]
            f = func(x)
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            # Fallback if no calls were made (should not happen)
            x_best = rng.uniform(-5, 5, dim)
            f_best = func(x_best)
            calls += 1

        # Stage 2: CMA-ES centered at the best grid point
        # Initialize CMA-ES parameters
        x_mean = x_best.copy()
        sigma = 1.0 # Initial step size
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # CMA-ES parameters
        mu = dim // 2 + 1  # Number of parents
        lambda_pop = 4 * mu # Population size
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim ** 1.7 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1) / (dim ** 1.7 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute C_sqrt update constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        while calls < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_pop):
                if time.time() - start_time > time_limit:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)
                
            if not ys:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update best
            if fs[0] < f_best:
                f_best = fs[0]
                x_best = ys[0].copy()
                
            # Take top mu individuals
            ys_mu = np.array(ys[:mu])
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * ys_mu, axis=0)
            
            # Update evolution paths
            y_sigma = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            # Update C
            h_sigma = (p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1 + 2 / (dim + 1)) * chi_n)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update covariance matrix
            C = (1 + c_1 * h_sigma * (h_sigma - 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            v = np.array([(y - x_mean) / sigma for y in ys_mu])
            C = C + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(v, v), axis=0)
            
            # Symmetrize and update square root
            C = (C + C.T) / 2
            # Simple square root update using eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            
            # Update mean
            x_mean = x_new
            
            # If population is stuck, restart with a larger sigma or new center
            # Simple stagnation check: if no improvement for 50 steps, reduce sigma further or restart
            # For now, rely on CMA-ES properties

        return x_best, f_best
# EVOLVE-BLOCK-END
