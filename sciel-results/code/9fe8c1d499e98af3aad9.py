"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        # CMA-ES parameters
        n = self.dim
        # Initial lambda (population size). For n=5, lambda=4n+2=22 is standard, 
        # but we have a tight budget. Let's use a smaller lambda to allow more generations.
        # lambda = 2n + 1 = 11? Or maybe even smaller to get more iterations.
        # Standard CMA-ES uses mu = lambda/2.
        # Let's try lambda = 10 (mu=5).
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
        weights /= np.sum(weights)
        
        # Variance of weighted sum of ranks (sigma_eff)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial mean
        # The problem is in [-5, 5]^5. Minimizer is in [-4, 4]^5.
        # Starting at 0 is a reasonable guess for centered distributions, 
        # but hidden shifts exist. 
        # However, CMA-ES is rotation invariant and shift invariant once it adapts.
        # A random start might be safer if the shift is large, but 0 is often good for BBOB.
        # Let's start at 0.
        x_mean = np.zeros(n)
        
        # Initial sigma
        # Box is [-5, 5]. Range is 10.
        # A good initial sigma is roughly (max - min) / 2.4? Or just 1.0?
        # If we start at 0, and optimum is in [-4,4], sigma=1.0 covers most of it.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Best so far
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Initial evaluation count: 1
        calls_made = 1
        
        # Helper for matrix square root and inverse square root
        # Using eigendecomposition for stability in low dimensions
        def update_cma(C, C_inv_sqrt):
            # C = C_inv_sqrt.T @ C_inv_sqrt
            # We maintain C and C_inv_sqrt
            pass

        # Main loop
        # We need to ensure we don't exceed budget (500) or time (290s)
        while calls_made < self.budget:
            if time.time() > deadline - 1:  # 1s safety margin for return
                break
                
            # Generate population
            # y = z * C_inv_sqrt
            # We need to generate z ~ N(0, I)
            
            # To speed up, we can pre-allocate arrays
            # But n=5 is tiny, overhead is negligible compared to func calls?
            # func calls are the bottleneck (black box).
            
            # Generate lambda samples
            # z_i ~ N(0, I)
            Z = np.random.randn(lambda_, n)
            
            # Transform to x space: x_i = x_mean + sigma * C_inv_sqrt @ z_i
            # C_inv_sqrt is n x n. Z is lambda_ x n.
            # X = x_mean + sigma * (C_inv_sqrt @ Z.T).T
            X = x_mean + sigma * (C_inv_sqrt @ Z.T).T
            
            # Clip to box? 
            # The problem states: "func(x) clips a finite length-5 vector to the box"
            # So we don't need to clip ourselves, but it might be good for internal bookkeeping?
            # No, let the func handle it. It saves us computation.
            
            # Evaluate
            F = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline - 1:
                    # Fill remaining with best known or break?
                    # If we break, we can't do the CMA update properly.
                    # It's better to break the outer loop entirely.
                    # But we are inside the for loop.
                    # Let's just break out of the for loop and then the while loop.
                    # We need to handle partial evaluation.
                    # Simplest: if time is up, stop evaluating, break outer loop.
                    F[i:] = np.inf
                    break
                F[i] = func(X[i])
                calls_made += 1
            
            if time.time() > deadline - 1:
                break
                
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Select best mu
            X_mu = X_sorted[:mu]
            
            # New mean
            x_new = np.sum(weights[:, np.newaxis] * X_mu, axis=0)
            
            # Update covariance matrix
            # First, update evolution paths
            
            # p_sigma update
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            # Note: (x_new - x_mean) / sigma is in z-space
            d_z = (x_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ d_z)
            
            # h_sigma check
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^{2(gen+1)}) < (1.4 + 2/(n+1)) * E||N(0,I)||
            # We don't track generations explicitly, but we can approximate.
            # For simplicity and speed, let's skip the h_sigma check or use a simplified version.
            # If we skip it, p_c update is always active.
            # Standard CMA-ES uses it to prevent p_c from growing too fast.
            # Let's implement a simplified check or just always update p_c.
            # Given n=5 and small budget, always updating p_c is likely fine.
            
            # p_c update
            # p_c = (1 - cc) * p_c + h_sigma * sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            # Using h_sigma = 1.0
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (C_inv_sqrt @ d_z)
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * sum w_i (y_i @ y_i.T)
            # y_i = (X_mu[i] - x_mean) / sigma
            Y = (X_mu - x_mean) / sigma
            
            # Sum of outer products
            sum_yy = np.zeros((n, n))
            for i in range(mu):
                sum_yy += weights[i] * np.outer(Y[i], Y[i])
                
            # Correction term for p_c
            # If h_sigma = 1, the (1 - h_sigma) term is 0.
            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * sum_yy
            
            # Update C_inv_sqrt
            # Eigendecomposition
            # C is symmetric positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative values to 0
            eigvals = np.clip(eigvals, 1e-12, None)
            
            # C_inv_sqrt = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ (np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)
            
            # Update sigma
            # sigma = sigma * exp(cs * (||p_sigma|| / sigma_eff - 1) / 5)
            # sigma_eff = sqrt(mu_eff)
            sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1) / 5)
            
            # Update mean
            x_mean = x_new
            
            # Optional: Re-center if mean is far out of bounds?
            # The problem clips inputs. If mean goes to 100, sigma might explode or shrink.
            # CMA-ES handles this usually, but clipping might cause issues.
            # Since func clips, gradients are 0 outside.
            # If we are outside, we might want to pull back.
            # But let's trust CMA-ES.

        return best_x, best_f
# EVOLVE-BLOCK-END
