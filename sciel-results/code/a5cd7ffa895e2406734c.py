"""Rewritten optimizer for BBOB f17 (Schaffers F7) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with restarts for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        total_calls = 0
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Number of parents mu
        mu = lam // 2
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial sigma
        sigma0 = 2.0
        
        # Restart logic
        # We restart if we run out of budget or if the algorithm stagnates
        # For Schaffers F7, restarts are helpful to escape local minima
        max_restarts = 10
        
        for restart_iter in range(max_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Initialize center
            if restart_iter == 0:
                center = np.zeros(dim)
            else:
                # Random restart within the box
                center = np.random.uniform(-5, 5, dim)
            
            # Check initial point
            f_center = func(center)
            total_calls += 1
            if f_center < best_f_global:
                best_f_global = f_center
                best_x_global = center.copy()
            
            # CMA-ES State
            x_mean = center.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # CMA-ES Loop
            # We run CMA-ES until stagnation or budget/time limit
            # Stagnation: if sigma * sqrt(trace(C)) becomes very small
            stagnation_threshold = 1e-6
            
            while total_calls < budget:
                if time.time() > deadline - 2.0:
                    break
                
                # Check for stagnation
                # If the step size is too small relative to the problem scale
                if sigma * np.sqrt(np.trace(C)) < stagnation_threshold:
                    break
                
                # Generate population
                # Cholesky factorization of C
                try:
                    Cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset C
                    C = np.eye(dim)
                    Cholesky = np.eye(dim)
                
                # Sample lambda points
                ys = np.random.randn(lam, dim)
                # Transform to x-space
                # x_i = x_mean + sigma * (Cholesky @ y_i)
                # We do this efficiently
                zs = ys @ Cholesky.T
                xs = np.array([x_mean + sigma * z for z in zs])
                
                # Evaluate population
                fs = np.zeros(lam)
                for i in range(lam):
                    if total_calls >= budget:
                        break
                    if time.time() > deadline - 2.0:
                        break
                    # Clip to box [-5, 5]
                    x_clipped = np.clip(xs[i], -5, 5)
                    fs[i] = func(x_clipped)
                    total_calls += 1
                    if fs[i] < best_f_global:
                        best_f_global = fs[i]
                        best_x_global = x_clipped.copy()
                
                # If we broke out of eval loop early, break CMA loop
                if total_calls >= budget or time.time() > deadline - 2.0:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                xs_sorted = xs[order]
                fs_sorted = fs[order]
                
                # Update mean
                x_new_mean = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # Update step size control paths
                pc = (1 - cs) * pc + cs * np.sqrt(mu_eff) * np.linalg.inv(Cholesky) @ (x_new_mean - x_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma
                
                # Update C
                delta_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - np.sqrt(dim)) * cs
                C = (1 + c1 * delta_sigma + c1 * (cmu - c1)) * C \
                    - c1 * np.outer(pc, pc) \
                    + cmu * np.sum(w[:mu][:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
                
                # Symmetrize C to avoid numerical drift
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - np.sqrt(dim)))
                
                # Update mean
                x_mean = x_new_mean
                
                # Check for convergence/stagnation
                # If the best fitness hasn't improved significantly in recent steps
                # Or if sigma is very small
                if sigma < 1e-8:
                    break
        
        # Ensure at least one call was made
        if total_calls == 0:
            x = np.zeros(dim)
            fx = func(x)
            best_x_global = x
            best_f_global = fx

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
