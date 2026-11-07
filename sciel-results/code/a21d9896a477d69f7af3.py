"""CMA-ES based optimizer for BBOB-style f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s limit

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        n_random = 30
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            try_eval(x)

        if best_x is None:
            x0 = np.zeros(dim)
            try_eval(x0)

        if best_x is None or calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES implementation
        # CMA-ES is a powerful derivative-free optimizer that adapts the search distribution
        # based on the history of successful steps.

        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        lambda_ = 4 * dim  # Population size
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalue decomposition for efficient updates
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Strategy parameters
        mu = lambda_ // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Damping factor for sigma update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Counter for stagnation
        no_improvement = 0
        max_no_improvement = 100
        
        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Generate population
            pop_x = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply transformation: x_mean + sigma * B * D * z
                x_new = x_mean + sigma * (eigvecs @ (eigvals * z))
                fx = try_eval(x_new)
                
                if fx is not None:
                    pop_x[i] = x_new
                    pop_f[i] = fx
            
            if i + 1 < lambda_:
                break  # Not enough evaluations completed
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Check for stagnation
            if no_improvement > max_no_improvement:
                # Restart with smaller sigma or break
                sigma *= 0.5
                no_improvement = 0
                if sigma < 1e-8:
                    break
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (pop_x[i] - x_mean)
            x_mean = x_mean + y_w
            
            # Update evolution paths
            y_sigma = (eigvecs.T @ (y_w / sigma))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (eigvecs @ y_sigma)
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + delta * chi_n^2 * C) + c_mu * sum(w_i * y_i * y_i.T)
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (np.dot(p_sigma, p_sigma) / chi_n**2 - 1) * np.eye(dim) * chi_n**2)
            
            for i in range(mu):
                y_i = (pop_x[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Check convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
