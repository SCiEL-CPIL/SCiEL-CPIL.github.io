"""Rewritten program using CMA-ES with multi-restart strategy for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES based optimizer with multi-restart strategy.
    CMA-ES adapts the covariance matrix to align with the basin orientation,
    which is critical for rotated Lunacek bi-Rastrigin functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_min = -5.0
        self.x_max = 5.0

    def _cma_es_step(self, func, x0, sigma0, budget_remaining, rng, deadline, best_x, best_y, calls_made):
        """
        Run CMA-ES for a limited number of calls.
        Returns: best_x, best_y, calls_made
        """
        dim = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + c_sigma
        # Covariance adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        # Step size
        sigma = sigma0
        # Mean
        mean = x0.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Cholesky of C
        try:
            C_sqrt_inv = np.linalg.cholesky(C)
            C_sqrt_inv = np.linalg.inv(C_sqrt_inv)
        except np.linalg.LinAlgError:
            C_sqrt_inv = np.eye(dim)
        
        # Counter for no improvement
        no_improve_count = 0
        best_local_x = best_x.copy()
        best_local_y = best_y
        
        while calls_made < self.budget and (self.budget - calls_made) > 0:
            if time.time() > deadline - 5:
                break
            
            # Generate population
            Y = rng.standard_normal((lambda_, dim))
            # Transform: x = mean + sigma * C_sqrt_inv @ y
            # Actually, we want x = mean + sigma * (C @ y) where y is standard normal
            # C_sqrt_inv is such that C_sqrt_inv @ C_sqrt_inv.T = C
            # So x = mean + sigma * (C_sqrt_inv @ y) gives covariance sigma^2 * C
            X = mean[None, :] + sigma * (Y @ C_sqrt_inv.T)
            
            # Clip to bounds
            X = np.clip(X, self.x_min, self.x_max)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_
            
            if time.time() > deadline - 5:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_local_y:
                best_local_y = F_sorted[0]
                best_local_x = X_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if best_local_y < best_y:
                best_y = best_local_y
                best_x = best_local_x.copy()
            
            # Check for convergence
            if no_improve_count > 30:
                break
            
            # Weighted mean of best mu solutions
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * \
                      C_sqrt_inv.T @ ((m_new - mean) / sigma)
            
            # h_sigma check
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim**2))
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lambda_))) < \
                      (1.4 + 2 / (dim + 1)) * chi_n
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - mean) / sigma
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (X_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_c * c_mu * delta_h - c_c - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definiteness
            try:
                C_sqrt_inv = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt_inv)
            except np.linalg.LinAlgError:
                # Add jitter
                C += 1e-6 * np.eye(dim)
                try:
                    C_sqrt_inv = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(C_sqrt_inv)
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)
            
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update mean
            mean = m_new
            
            # If mean goes out of bounds, pull it back
            mean = np.clip(mean, self.x_min, self.x_max)
        
        return best_x, best_y, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin (300s limit, 10s buffer)
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0
        
        # Multi-restart strategy
        # Lunacek bi-Rastrigin has two basins. We try multiple starting points.
        # Budget allocation: split budget across restarts
        # First restart gets more budget, subsequent ones get less
        
        # Initial exploration: a few random points to get a sense of the landscape
        n_init = 10
        X_init = rng.uniform(self.x_min, self.x_max, size=(n_init, self.dim))
        Y_init = np.array([func(x) for x in X_init])
        calls_made += n_init
        
        best_idx = np.argmin(Y_init)
        best_x = X_init[best_idx].copy()
        best_y = Y_init[best_idx]
        
        # Define restart points
        # Mix of random points and points near the best found
        restarts = []
        
        # First, use the best initial point
        restarts.append((best_x.copy(), 1.0))
        
        # Add some diverse starting points
        for _ in range(5):
            # Mix: some near best, some random
            if rng.random() < 0.5:
                # Near best
                x_r = best_x + rng.normal(0, 1.5, size=self.dim)
            else:
                # Random
                x_r = rng.uniform(self.x_min, self.x_max, size=self.dim)
            x_r = np.clip(x_r, self.x_min, self.x_max)
            restarts.append((x_r, 0.8))
        
        # Budget allocation for restarts
        # First restart gets 50% of remaining budget, rest split among others
        remaining_budget = self.budget - calls_made
        
        # Sort restarts by initial distance to best (closer first)
        # Actually, let's just use them in order
        
        # Allocate budget: first restart gets half, rest split evenly
        budget_allocs = []
        first_alloc = int(remaining_budget * 0.5)
        budget_allocs.append(first_alloc)
        rest_budget = remaining_budget - first_alloc
        if len(restarts) > 1:
            per_restart = rest_budget // (len(restarts) - 1)
            for i in range(1, len(restarts)):
                budget_allocs.append(per_restart)
        
        # Run CMA-ES from each restart
        for i, (x0, sigma_factor) in enumerate(restarts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            
            alloc = budget_allocs[i] if i < len(budget_allocs) else 0
            if alloc <= 0:
                break
            
            # Don't exceed total budget
            alloc = min(alloc, self.budget - calls_made)
            if alloc <= 0:
                break
            
            sigma0 = 1.0 * sigma_factor  # Initial step size
            # If we're not at the best point, use a larger sigma
            if i > 0:
                sigma0 = 1.5 * sigma_factor
            
            best_x, best_y, calls_made = self._cma_es_step(
                func, x0, sigma0, alloc, rng, deadline, best_x, best_y, calls_made
            )
        
        # Final polish: if we have budget left, do local search around best
        remaining = self.budget - calls_made
        if remaining > 0 and time.time() < deadline - 10:
            # Simple local polish
            radius = 0.5
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                x_polish = best_x + rng.uniform(-radius, radius, size=self.dim)
                x_polish = np.clip(x_polish, self.x_min, self.x_max)
                y_polish = func(x_polish)
                calls_made += 1
                if y_polish < best_y:
                    best_y = y_polish
                    best_x = x_polish.copy()
                radius *= 0.95
        
        return best_x, best_y

# EVOLVE-BLOCK-END
