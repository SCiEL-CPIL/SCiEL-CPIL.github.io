import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lambda_pop = 4 + int(3 * np.log(dim))  # For dim=5, lambda = 4 + 3*1.609 = 8.8 -> 8? No, int(3*1.609)=4, so 4+4=8. 
        # Let's use a slightly larger population for better exploration in 5D.
        # Standard is often 4+3ln(d). For d=5, 4+4=8. Let's use 10 or 12 for robustness.
        lambda_pop = 10 
        mu = lambda_pop // 2  # Number of best solutions used for update
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Initial values
        # Start at the center of the box
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, large enough to explore [-5, 5]
        C = np.eye(dim)
        DC = np.linalg.cholesky(C)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        dampc = (4 * mu_eff - 2) / (dim + 4)
        
        # Best solution tracking
        best_x = x_mean.copy()
        best_fx = float('inf')
        calls = 0
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal calls, best_x, best_fx
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx
        
        # Initial evaluation
        fx = evaluate(x_mean)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.array([np.random.standard_normal(dim) for _ in range(lambda_pop)])
            # Transform using Cholesky factor: x_i = x_mean + sigma * DC @ y_i
            # Note: DC is lower triangular such that C = DC @ DC.T
            # So C^(1/2) = DC
            zs = np.array([DC @ y for y in ys])
            xs = np.array([x_mean + sigma * z for z in zs])
            
            # Clip to box
            xs = np.clip(xs, -5, 5)
            
            # Evaluate population
            fs = np.array([evaluate(xi) for xi in xs])
            
            if calls >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            xs_sorted = xs[order]
            
            # Update mean
            x_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # Update evolution path ps
            # C^(-1/2) = DC^{-T} @ DC^{-1}
            try:
                DC_inv = np.linalg.inv(DC)
                C_inv_half = DC_inv.T @ DC_inv
            except np.linalg.LinAlgError:
                C_inv_half = np.eye(dim)
                
            diff = x_new - x_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C_inv_half @ diff)
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            # Expected norm of standard normal vector in dim dimensions
            E_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(cs / damps * (norm_ps / E_norm - 1))
            
            # Update evolution path pc and covariance C
            if norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (calls - lambda_pop) / (dim * damps))) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - (1 - cs) ** 2) / (1 - (1 - cs) ** (2 * (calls - lambda_pop) / (dim * damps)))):
                pc = (1 - c1) * pc + c1 / sigma * diff
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum(w[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]) * w[:, None], axis=0)
            else:
                C = (1 - c1 - cmu) * C + cmu * np.sum(w[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]) * w[:, None], axis=0)
                
            # Regularize C to ensure positive definite
            C = (C + C.T) / 2
            # Add small identity to prevent singularity
            C += 1e-12 * np.eye(dim)
            
            # Update Cholesky
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity or use previous DC
                # This can happen if C becomes ill-conditioned
                DC = np.eye(dim)
                C = np.eye(dim)
                # Reset paths if we had to reset C
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                sigma = max(sigma, 0.1)
                
            x_mean = x_new
            
            # If sigma becomes too small or too large, adjust
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        return best_x, best_fx
