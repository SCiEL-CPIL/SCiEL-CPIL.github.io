import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return np.inf
            calls_used += 1
            return func(x)
        
        dim = self.dim
        budget = self.budget
        
        # Stage 1: Sparse Global Exploration (Latin Hypercube-like)
        # We use a small number of points to identify a promising basin.
        n_scatter = 20
        scatter_points = []
        
        # Simple Latin Hypercube Sampling implementation for diversity
        # Divide each dimension into n_scatter bins, pick one random point per bin
        # This ensures coverage of the domain [-5, 5]^dim
        if n_scatter > 0:
            for i in range(n_scatter):
                x = np.zeros(dim)
                for d in range(dim):
                    # Bin index for this dimension is effectively i % n_scatter
                    # But to decorrelate, we can just use random uniform 
                    # or a stratified random sample.
                    # Let's use stratified random sampling for better coverage
                    # Bin i for dim d: [i/n, (i+1)/n] mapped to [-5, 5]
                    # Actually, simple random is often sufficient for N=20 in D=5.
                    # Let's just use random points but ensure we don't start at 0 always.
                    pass
            
            # Simpler robust approach: 10 random points + 10 points on a coarse grid
            # Or just 20 random points.
            # Let's use 20 random points in [-4, 4] to avoid boundaries initially.
            for _ in range(n_scatter):
                if calls_used >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-4.0, 4.0, dim)
                fx = eval_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        if best_x is None:
            # Fallback if no scatter points evaluated
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        # Stage 2: CMA-ES Local Refinement
        # Initialize CMA-ES from the best point found
        x0 = best_x.copy()
        sigma = 0.5  # Smaller initial step size as we are likely in a basin
        
        # CMA-ES parameters
        lam = 12
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        def mat_sqrt(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if calls_used + 1 > budget or time.time() > deadline:
                    break
                zi = rng.standard_normal(dim)
                yi = x0 + sigma * (C_sqrt @ zi)
                yi = np.clip(yi, -5, 5)
                yi = yi.copy()
                ys.append(yi)
            
            if len(ys) == 0:
                break
            
            # Evaluate population
            fvals = []
            for y in ys:
                if calls_used >= budget or time.time() > deadline:
                    break
                f = eval_func(y)
                fvals.append(f)
            
            if len(fvals) < 2:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Update mean (recombination)
            x_mean_old = x0.copy()
            x0 = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update evolution paths
            C_inv_sqrt = mat_sqrt(np.linalg.inv(C))
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x0 - x_mean_old) / sigma)
            
            # h_sigma calculation simplified for stability
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (1)))) < (1 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma)) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x0 - x_mean_old) / sigma
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(ys_sorted[i] - x_mean_old, ys_sorted[i] - x_mean_old) for i in range(mu))
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * C) + c_mu * rank_mu
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_sqrt
            C_sqrt = mat_sqrt(C)
            
            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Convergence check
            if sigma < 1e-8:
                break
            if np.all(np.abs(C - np.eye(dim) * np.trace(C) / dim) < 1e-10):
                break
        
        return best_x, best_f
