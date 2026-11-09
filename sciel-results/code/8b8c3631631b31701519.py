"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to identify basins
        # Use 30 random points in [-4, 4]^5
        num_initial = 30
        initial_points = []
        for i in range(num_initial):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=self.dim)
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x.copy()))
        
        # Sort by fitness and take top K basins
        initial_points.sort(key=lambda item: item[0])
        top_k = min(5, len(initial_points))
        
        # Phase 2: CMA-ES from top basins
        # CMA-ES parameters
        pop_size = 10  # lambda
        sigma_min = 0.005
        sigma_max = 5.0
        max_local_steps = 40  # Max generations per CMA-ES instance
        
        for idx in range(top_k):
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Start from the idx-th best initial point
            f_start, x_start = initial_points[idx]
            x_mean = x_start.copy()
            sigma = 0.5
            
            # CMA-ES state
            C = np.eye(self.dim)
            C_sqrt = np.eye(self.dim)
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Adaptation parameters
            mu = pop_size // 2  # Number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((self.dim + 2) ** 2 + mu_eff))
            
            stall_count = 0
            best_in_cma = f_start
            for gen in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(self.dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    offspring.append((fy, y))
                
                if len(offspring) < pop_size:
                    break
                
                # Sort offspring by fitness
                offspring.sort(key=lambda item: item[0])
                
                # Select best mu individuals
                selected = offspring[:mu]
                
                # Track best in this CMA-ES run
                if selected[0][0] < best_in_cma:
                    best_in_cma = selected[0][0]
                
                # Check for stall
                if best_in_cma >= f_start * 0.999 and gen > 5:
                    stall_count += 1
                    if stall_count >= 10:
                        break
                else:
                    stall_count = 0
                
                # Update mean
                new_mean = sum(weights[i] * selected[i][1] for i in range(mu))
                
                # Update covariance matrix
                y_vec = (new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1 + 3 / self.dim) * np.sqrt(1 - (1 - c_sigma) ** 2)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                
                delta_h = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) < (1 + 2 / self.dim) * np.sqrt(1 - (1 - c_c) ** 2)
                
                C = ((1 + c_1 + c_mu - c_1) * C +
                    c_1 * (np.outer(p_c, p_c) - C) +
                    c_mu * sum(weights[i] * np.outer(selected[i][1] - x_mean, selected[i][1] - x_mean) for i in range(mu)) / sigma ** 2)
                
                # Ensure positive semi-definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Update x_mean
                x_mean = new_mean
                
                # Update C_sqrt via Cholesky decomposition (approximate)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Fallback to eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Phase 3: Final polish with local (1+1)-ES
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for step in range(50):
                if calls >= self.budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 1.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.9, 0.01)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
