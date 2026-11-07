"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 290s)
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls_made = 0
        global_best_x = None
        global_best_f = np.inf

        # Wrapper to count calls, enforce time/budget, and clip
        def wrapped_func(x):
            nonlocal calls_made, global_best_f, global_x
            if time.time() > deadline:
                return global_best_f
            if calls_made >= total_budget:
                return global_best_f
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_made += 1
            
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x.copy()
                
            return f_val

        # CMA-ES Parameters
        # We will run multiple restarts.
        # Let's split the budget among a few restarts.
        # Weierstrass is multimodal. A single start might miss the global basin.
        # However, CMA-ES is expensive per iteration (population size N ~ 4 + 3*ln(n)).
        # For dim=5, N = 4 + 3*ln(5) ~ 9.
        # Each iteration costs N function calls.
        # If we use 500 calls, we can do ~55 iterations per restart.
        # Let's try 3 restarts to cover different regions.
        
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        
        # Define initial centers for restarts.
        # Since the minimizer is in [-4, 4]^5, we can spread starts.
        # Start 0: Origin (0,0,0,0,0)
        # Start 1: (1, 1, 1, 1, 1) 
        # Start 2: (-1, -1, -1, -1, -1)
        # Or use a small Sobol set for initial centers.
        
        centers = [
            np.zeros(dim),
            np.ones(dim) * 1.5,
            np.ones(dim) * -1.5
        ]
        
        # If we have more budget, we could add more, but 3 is a good start.
        # If dim was larger, we'd need more. For dim=5, 3 is reasonable.
        
        # Initial sigma (step size).
        # Weierstrass has features at various scales.
        # A sigma of 1.0 is a good start for [-5,5] box.
        initial_sigma = 1.0
        
        for i in range(n_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
                
            # Remaining budget for this restart
            remaining_in_restart = total_budget - calls_made
            # Cap the restart budget to what we planned, but don't exceed remaining
            local_budget = min(budget_per_restart, remaining_in_restart)
            
            if local_budget < 20: # Not enough for a meaningful CMA-ES run
                break
                
            x_center = centers[i % len(centers)].copy()
            
            # CMA-ES implementation
            # Standard CMA-ES parameters
            N = 4 + int(3 * np.log(dim))  # Population size
            if N < 4: N = 4
            
            # Initialize
            mean = x_center
            sigma = initial_sigma
            C = np.eye(dim)
            C_inv = np.eye(dim)
            
            # Step sizes for C and sigma updates
            mu = N // 2  # Number of best solutions to use for update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_c_inv = 2 * (dim + 2 - mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            
            # Pre-compute
            # We need to generate N samples from N(mean, sigma^2 * C)
            
            # Loop until budget for this restart is exhausted
            while calls_made < total_budget and (total_budget - calls_made) >= N:
                if time.time() > deadline:
                    break
                
                # 1. Sample N points
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes ill-conditioned
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Standard normal samples
                Z = np.random.randn(N, dim)
                # Transform
                X = np.array([mean + sigma * (Z[i] @ L.T) for i in range(N)])
                
                # 2. Evaluate
                F = np.array([wrapped_func(x) for x in X])
                
                # Check if we ran out of budget mid-evaluation
                # wrapped_func handles the counting, but we should stop if we hit the limit
                if calls_made >= total_budget:
                    break
                    
                # 3. Sort by fitness
                order = np.argsort(F)
                X_sorted = X[order]
                F_sorted = F[order]
                
                # 4. Update mean
                mean_old = mean
                mean = np.sum([weights[i] * X_sorted[i] for i in range(mu)], axis=0)
                
                # 5. Update path variables
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (sigma * (mean - mean_old)))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ (mean - mean_old))
                
                # 6. Update Covariance Matrix
                # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T) + cmu * sum(weights * (y_i @ y_i.T))
                c1 = 2 / ((dim + 1)**2 + 10)
                cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff)
                
                Y = np.array([(X_sorted[i] - mean_old) / sigma for i in range(mu)])
                C_update = np.sum([weights[i] * np.outer(Y[i], Y[i]) for i in range(mu)], axis=0)
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * C_update
                
                # 7. Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # 8. Update C_inv
                # To save computation, we can re-compute C_inv every 10 iterations or when needed
                # For dim=5, it's cheap.
                C_inv = np.linalg.inv(C)
                L_inv = np.linalg.inv(L)
                
                # Optional: Early stop if convergence is very slow
                # But we rely on the outer restart loop.
                
                # Check if we should break the inner loop to save time for next restart?
                # The condition (total_budget - calls_made) >= N handles this.

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
