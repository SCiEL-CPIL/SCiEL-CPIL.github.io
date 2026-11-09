"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with PCA-initialized covariance for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # 4 + 3*ln(5) ≈ 8
        pop_size = max(pop_size, 2)
        
        # Initial point: center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # PCA initialization: use a small set of evaluations to estimate the landscape
        # For Schaffers F7 (ill-conditioned funnel), the landscape has a strong directional structure.
        # We'll do a quick PCA-based initialization using a few random points.
        
        # First, let's do a brief random search to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        # Quick exploration to estimate covariance structure
        n_explore = min(10, budget - 1)
        explore_x = []
        explore_f = []
        
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            xi = rng.uniform(-5, 5, dim)
            fi = func(xi)
            calls += 1
            explore_x.append(xi)
            explore_f.append(fi)
        
        explore_x = np.array(explore_x)
        explore_f = np.array(explore_f)
        
        # Update best
        for i in range(len(explore_f)):
            if explore_f[i] < best_f:
                best_f = explore_f[i]
                best_x = explore_x[i].copy()
        
        # Estimate covariance from exploration points
        # Center the points
        mean_explore = explore_x.mean(axis=0)
        centered = explore_x - mean_explore
        cov = np.cov(centered, rowvar=False)
        
        # Ensure covariance is positive definite
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Initialize CMA-ES
        sigma = 1.0
        c_sigma = (pop_size + 2) / (pop_size + 5 + 2 * dim / pop_size)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights @ weights)
        c_c = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        step_size_sigma = 0
        step_size_c = 0
        
        # Transform: x_new = x + sigma * B @ z where B^T B = C
        # Use Cholesky decomposition
        try:
            B = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            B = np.eye(dim)
        
        # CMA-ES main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            zs = rng.standard_normal((pop_size, dim))
            ys = [x + sigma * (B @ z.T).T for z in zs]
            ys = np.array([np.clip(y, -5, 5) for y in ys])
            
            # Evaluate
            fs = []
            for y in ys:
                if time.time() > deadline:
                    break
                fy = func(y)
                fs.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(fs) < pop_size:
                break
            
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            x_new = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update step size (CSA)
            y_mean = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - pop_size))) <
                       (1 + 2 / (dim + 1)) * chi_n)
            h_sigma = float(h_sigma)
            
            delta_d = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = np.outer(
                (x_new - x) / sigma, (x_new - x) / sigma
            )
            rank_one = np.outer(p_c, p_c)
            
            cov = (1 + c_c * (2 - c_c) - mu_eff * c_sigma * c_sigma / d_sigma) * cov + \
                  c_sigma * (rank_mu - delta_d * rank_one)
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-12)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            try:
                B = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                B = np.eye(dim)
            
            x = x_new
        
        # Final polish: do a few local refinements around the best point
        if calls < budget:
            polish_budget = min(20, budget - calls)
            sigma_polish = sigma * 0.1
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_polish *= 0.8
                else:
                    sigma_polish *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
