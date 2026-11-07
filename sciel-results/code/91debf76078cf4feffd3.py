"""Surrogate-assisted optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget

        # Initial population
        rng = np.random.default_rng(42)
        
        # Start near center with some spread
        x = np.zeros(dim)
        sigma = 2.5
        
        # CMA-ES parameters
        # Standard CMA-ES parameters
        lam = 4 * (int(1 + 3 * dim) // 4)  # population size
        if lam < 4:
            lam = 4
            
        # Step size control
        c_sigma = (4 + dim / dim) / (dim + 4 + 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + c_sigma
        
        # Mean update
        c_m = (4 + lam / dim) / (dim + 4 + 3 * lam)
        
        # Covariance matrix adaptation
        c_c = (4 + lam / dim) / (dim + 9 + 3 * lam)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / ((dim + 2) * (lam - mu_eff + 2)) - c_1))
        
        # Weights
        w = np.log(np.arange(1, lam + 1))
        w = w[w > 0]
        if len(w) == 0:
            w = np.ones(lam)
        w = w / w.sum()
        mu = int(lam // 2)
        w = w[:mu]
        w = w / w.sum()
        mu_eff = 1 / (w ** 2).sum()
        
        # Recompute c_mu
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / ((dim + 2) * (lam - mu_eff + 2)) - c_1))
        
        # Initial covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Evolving paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky decomposition
        def cholesky(matrix):
            try:
                return np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                # Add small regularization
                return np.linalg.cholesky(matrix + 1e-6 * np.eye(matrix.shape[0]))
        
        B = cholesky(C)
        D = np.sqrt(np.diag(C))
        inv_B = np.linalg.inv(B)
        D_inv = 1.0 / D
        
        # Initial population
        def generate_population():
            pop = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                pop.append(x + sigma * (B @ (D * z)))
            return np.array(pop)
        
        # Evaluate initial population
        pop = generate_population()
        fvals = np.array([func(p) for p in pop])
        
        # Sort by fitness
        idx = np.argsort(fvals)
        pop = pop[idx]
        fvals = fvals[idx]
        
        # Initial mean update
        w_sum = w.sum()
        x_new = (w * pop[:mu]).T @ np.ones((mu, 1)).flatten()
        # Actually, weighted average
        x_new = np.sum(w[:, np.newaxis] * pop[:mu], axis=0)
        
        # Track best
        best_x = pop[0].copy()
        best_f = fvals[0]
        
        calls_made = lam
        
        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Update paths
            sigma_old = sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_B @ (D_inv @ ((x_new - x) / sigma)))
            
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt((lam + 1 - 2) * (1 - (1 - c_sigma) ** 2))) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * ((x_new - x) / sigma)))
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0)
            
            C = (1 + c_sigma * c_mu * mu_eff - c_1 - c_sigma) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2  # Symmetrize
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x = x_new
            
            # Regenerate population
            pop = generate_population()
            
            # Evaluate
            fvals = np.array([func(p) for p in pop])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
            
            # Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # New mean
            x_new = np.sum(w[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update Cholesky
            B = cholesky(C)
            D = np.sqrt(np.maximum(np.diag(C), 1e-10))
            inv_B = np.linalg.inv(B)
            D_inv = 1.0 / D
        
        return best_x, best_f
# EVOLVE-BLOCK-END
