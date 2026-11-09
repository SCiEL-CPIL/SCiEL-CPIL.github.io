"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Solver using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with adaptive restarts.
    
    Strategy:
    1. Run CMA-ES with a population size and step size adapted to the ill-conditioned nature.
    2. Monitor convergence and restart with adjusted parameters if stagnation is detected.
    3. Use a deadline guard to ensure we finish before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES parameters
        # Initial step size
        sigma0 = 1.5
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        lambda_ = max(lambda_, dim + 1)

        # Initial center: use a random point or center
        # We'll start near the center but with some randomization
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-2, 2, size=dim)

        # Evaluate initial point
        counting_func(x0)

        # CMA-ES implementation
        # State variables
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)  # Covariance matrix
        C_inv_sqrt = np.eye(dim)  # Use inverse sqrt for sampling
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES parameters (standard)
        mu = lambda_ // 2
        weights = np.zeros(mu)
        weights[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        cc = (4 + lambda_ / (dim + 4)) / ((dim + 4) + 4 + lambda_ / (dim + 4))
        cs = (2 + 2 / (dim + 1)) / (dim + 2 + lambda_ / (dim + 1))
        cmu = min(1 - cs, 2 / ((dim + 1.3)**2 + lambda_))
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Track convergence
        no_improve = 0
        patience = 25  # Number of iterations without improvement before considering restart
        max_restarts = 10
        restart_count = 0

        while calls < budget and time.time() < deadline:
            # Store old mean for path updates
            x_mean_old = x_mean.copy()
            
            # Generate population
            pop = np.zeros((lambda_, dim))
            f_values = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from multivariate normal using C_inv_sqrt
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (C_inv_sqrt.T @ z)
                f_values[i] = counting_func(x_i)
                pop[i] = np.clip(x_i, -5, 5)

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(f_values)
            pop_sorted = pop[order]
            f_sorted = f_values[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Weighted recombination
            x_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Update paths (using old mean)
            delta = (x_new - x_mean_old) / sigma
            p_sigma = (1 - cs) * p_sigma + cs * delta
            p_c = (1 - cc) * p_c + cc * delta

            # Update step size
            sigma *= np.exp(cs / chiN * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update covariance matrix
            # Rank-one update
            v = delta
            C = (1 + cmu * (mueff - 1) - cmu) * C + cmu * np.outer(v, v)
            
            # Regularize and compute inverse square root
            C_reg = C + 1e-10 * np.eye(dim)
            try:
                eigvals, eigvecs = np.linalg.eigh(C_reg)
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)

            # Update mean
            x_mean = x_new

            # Check for convergence
            if no_improve >= patience:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart with smaller step size and reset covariance
                sigma *= 0.7
                no_improve = 0
                x_mean = best_x.copy()
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        # Final polish phase using Nelder-Mead on remaining budget
        if calls < budget and time.time() < deadline:
            from scipy.optimize import minimize
            
            def polish_func(x):
                return counting_func(x)
            
            # Use Nelder-Mead for local refinement
            try:
                result = minimize(polish_func, best_x, method='Nelder-Mead', 
                                 options={'maxiter': min(50, budget - calls), 
                                          'xatol': 1e-6, 'fatol': 1e-10})
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
