"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    
    This is a fundamentally different approach from the (1+1)-ES used in the parent.
    CMA-ES adapts the full covariance matrix of the search distribution, allowing it to
    learn the anisotropic structure of the landscape. For f14 (Different powers), which
    has different scales along different axes, this is particularly powerful.
    
    Key advantages over (1+1)-ES:
    1. Adapts to the shape of the valley/basin, not just its direction
    2. Better sample efficiency in high-dimensional anisotropic landscapes
    3. No need for manual restarts - the covariance adaptation handles exploration
    
    We use a standard CMA-ES implementation with:
    - Population size lambda = 4 + floor(3*log(dim))
    - Standard CMA-ES update rules for mean, sigma, and covariance
    - Boundary handling via reflection
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        rng = np.random.default_rng(42)

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial parameters
        x_mean = np.zeros(dim)  # Start at center
        sigma = 2.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        p_sigma = np.zeros(dim)  # Evolution path for sigma
        p_c = np.zeros(dim)  # Evolution path for C
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Precompute
        inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C).T)
        
        best_x = x_mean.copy()
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return x_clipped, f

        # Initial evaluation
        x_mean, fx = evaluate(x_mean)
        if check_time():
            return best_x, best_f

        while not check_time():
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Use current inverse sqrt C for sampling
                try:
                    # Approximate: use eigenvalue decomposition for stability
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    y = eigvecs @ (np.sqrt(eigvals) * z)
                    x = x_mean + sigma * y
                except np.linalg.LinAlgError:
                    x = x_mean + sigma * z
                
                x, fx = evaluate(x)
                offspring.append(x)
                offspring_f.append(fx)
                
                if check_time():
                    break

            if len(offspring) < mu:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f[:mu * 2])[:mu]
            y_selected = np.array([offspring[i] for i in indices])
            f_selected = np.array([offspring_f[i] for i in indices])

            # Weighted mean of selected
            x_new = np.sum(weights[:, None] * y_selected, axis=0)

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                inv_sqrt_C @ (x_new - x_mean) / sigma
            )
            
            # h_sigma: check if p_sigma is consistent
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - lambda_)))) \
                      < (1.4 + 2 / (dim + 1)) * chi_N

            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                inv_sqrt_C @ (x_new - x_mean) / sigma
            )

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * np.outer(y_selected, y_selected), axis=0) \
                      - np.outer(x_mean, x_mean)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * (rank_one - (1 - delta_h) * np.eye(dim)) \
                + c_mu * (rank_mu / sigma ** 2 - np.eye(dim))

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            x_mean = x_new

            # Update inverse sqrt C periodically
            if calls_used % (lambda_ * 5) < lambda_:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    inv_sqrt_C = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    pass

            # Boundary handling: if we hit boundaries frequently, adapt
            # Check if many points were clipped
            clipped_count = sum(1 for x in offspring if not np.all((x >= -5) & (x <= 5)))
            if clipped_count > lambda_ // 2:
                sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
