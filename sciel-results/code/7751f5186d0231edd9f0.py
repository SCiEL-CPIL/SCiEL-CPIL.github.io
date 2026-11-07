"""Optimized CMA-ES for BBOB-style f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_counter = [0]

        def check_time():
            return time.time() < deadline

        def check_budget():
            return calls_counter[0] < budget

        def eval_point(x):
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good default.
        lambda_ = 10
        mu = lambda_ // 2  # Number of weights
        
        # Weights for recombination
        # Standard CMA-ES weights: proportional to log ranks
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        
        # Variance-effectiveness parameter
        mu_eff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        # Start with a moderate step size. The box is [-5, 5].
        # The optimum is in [-4, 4]. A step size of 2.0 covers the space well initially.
        sigma = 2.0
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Helper to update square root decomposition
        def update_C_sqrt():
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            # Use SVD for stability, though Cholesky is faster if PD.
            # C is PD.
            # We need C^{1/2} and C^{-1/2}
            # Standard CMA-ES uses a specific update rule for C_sqrt to avoid recomputing SVD every step.
            # However, for dim=5 and ~50 generations, SVD is fast enough.
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            C_sqrt_inv = U @ np.diag(1.0 / np.sqrt(s)) @ Vt
            C_inv = U @ np.diag(1.0 / s) @ Vt

        update_C_sqrt()

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial population
        # We will generate lambda samples each generation.
        # Total calls: 500. Lambda=10 -> 50 generations.
        
        # To improve initial exploration, we can bias the initial mean or use a larger sigma.
        # Let's stick to standard CMA-ES but ensure the initial sigma is large enough.
        
        while check_time() and check_budget():
            # Generate population
            Y = rng.multivariate_normal(np.zeros(dim), C, size=lambda_)
            # Transform to X
            X = np.array([x_mean + sigma * C_sqrt @ y for y in Y])
            
            # Evaluate
            F = np.array([eval_point(x) for x in X])
            
            # Check if we ran out of budget mid-evaluation
            if not check_budget():
                break
                
            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            # (Already updated in eval_point, but X[0] is the best)
            
            # Update mean
            x_new_mean = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new_mean - x_mean) / sigma
            y_sigma = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_sigma)
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * lambda_)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update Covariance Matrix
            # Delta_h = (2 - h_sigma) * c_c * (dim + 1) - 1
            delta_h = (2 - h_sigma) * c_c * (dim + 1) - 1
            
            # Rank-one update
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * p_c p_c^T + c_mu * sum w_i (y_i y_i^T)
            # Where y_i = (x_i - x_old_mean) / sigma
            Y_sorted = (X[:mu] - x_mean) / sigma
            rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', w, Y_sorted, Y_sorted), axis=0)
            
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Prevent sigma from becoming too large or small
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update square root decomposition
            update_C_sqrt()
            
            # Update mean
            x_mean = x_new_mean
            
            # Convergence check
            if sigma < 1e-12:
                break
                
            # Check if progress is negligible
            # If the best fitness hasn't improved in a while, we might be stuck.
            # But CMA-ES handles this well.
            
            # If the spread of the population is very small, stop.
            # This is implicitly handled by sigma shrinking.

        return best_x, best_f
# EVOLVE-BLOCK-END
